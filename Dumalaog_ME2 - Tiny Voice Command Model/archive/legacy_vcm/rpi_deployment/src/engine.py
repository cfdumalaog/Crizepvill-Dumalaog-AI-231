"""
Real-Time Sliding Window Streaming Inference Engine for Tiny VCM.
Manages continuous circular audio buffering, sliding-window feature extraction,
confidence thresholding, debounce holdoff, and command dispatch.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Callable, List

from .config import (
    NUM_SAMPLES,
    SAMPLE_RATE,
    CONFIDENCE_THRESHOLD,
    DETECTION_HOLDOFF_SEC,
    COMMAND_CLASSES,
    IDX_TO_CLASS,
    WAKE_WORD_CLASS,
    WAKE_TIMEOUT_SEC,
    WAKE_CONFIDENCE
)
from .audio import LogMelFrontend, AudioRingBuffer, EnergyVAD, AdaptiveEndpointer
from .actions import SmartDeviceController
from .tts import VOICE_ENGINE


class StreamingVCMEngine:
    """
    Real-Time Sliding Window Voice Command Recognition Engine with Wake-Word State Machine.
    Faithfully replicates the multi-stage cascaded architecture of Apple Siri & Amazon Alexa:
    - Stage 1: Ultra-low-power Acoustic Front-End & Energy VAD (bypasses inference during room silence).
    - Stage 2: Sliding Window Keyphrase Spotting (KWS) for "Hi Dandan" / "Hello Dandan".
    - Stage 3: Compound One-Shot ("One-Breath") Utterance Pipeline ("Hi Dandan play music" in one sentence).
    - Stage 4: Adaptive Acoustic Endpointing (instant command dispatch upon speech cessation without arbitrary delay).
    - Stage 5: Multi-Modal Spoken & Instant Synthesized Earcon Audio Feedback (rising/falling dual-tones).
    """
    STATE_STANDBY = "STANDBY"
    STATE_LISTENING = "LISTENING"

    def __init__(
        self,
        model: torch.nn.Module,
        controller: SmartDeviceController,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        wake_confidence: float = WAKE_CONFIDENCE,
        wake_timeout_sec: float = WAKE_TIMEOUT_SEC,
        holdoff_seconds: float = DETECTION_HOLDOFF_SEC,
        device: str = "cpu",
        require_wake_word: bool = True
    ):
        self.model = model.to(device)
        self.model.eval()
        self.controller = controller
        self.confidence_threshold = confidence_threshold
        self.wake_confidence = wake_confidence
        self.wake_timeout_sec = wake_timeout_sec
        self.holdoff_seconds = holdoff_seconds
        self.device = device
        self.require_wake_word = require_wake_word

        self.frontend = LogMelFrontend().to(device)
        self.ring_buffer = AudioRingBuffer(capacity_samples=NUM_SAMPLES)
        
        # Siri/Alexa Stage 1 Front-End: Energy VAD and Dynamic Endpointer
        self.vad = EnergyVAD(sample_rate=SAMPLE_RATE)
        self.endpointer = AdaptiveEndpointer(sample_rate=SAMPLE_RATE)
        self.last_audio_energy = 0.0
        self.last_snr = 0.0
        self.is_speech_active = False

        # State tracking
        self.state = self.STATE_STANDBY if require_wake_word else self.STATE_LISTENING
        self.wake_time = 0.0
        self.last_detection_time = 0.0
        self.last_detected_command = None

        # Statistics
        self.total_frames_processed = 0
        self.total_commands_triggered = 0
        self.total_wakes_triggered = 0
        self.compound_commands_triggered = 0
        self.vad_silence_bypasses = 0
        self.latencies_ms = []

        # Latest forward pass cache
        self.last_mel_spec = None
        self.last_top3 = {}
        self.last_predicted_command = None
        self.last_confidence = 0.0

    def set_require_wake_word(self, require: bool):
        """Toggle whether wake word is required to accept commands."""
        if self.require_wake_word != require:
            self.require_wake_word = require
            self.endpointer.reset()
            if not require:
                self.state = self.STATE_LISTENING
            else:
                self.state = self.STATE_STANDBY

    def check_timeout(self) -> Optional[Dict[str, Any]]:
        """Checks if active listening window has expired."""
        if self.state == self.STATE_LISTENING and self.require_wake_word:
            if time.time() - self.wake_time > self.wake_timeout_sec:
                self.state = self.STATE_STANDBY
                self.endpointer.reset()
                action_msg = self.controller.trigger_timeout()
                VOICE_ENGINE.play_earcon("cancel")
                return {
                    "event": "TIMEOUT",
                    "state": self.state,
                    "action_message": action_msg,
                    "timestamp": time.time()
                }
        return None

    def feed_audio_chunk(self, chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Feeds a chunk of 16kHz PCM audio samples into the ring buffer.
        Executes VAD gating, dynamic endpointing, and sliding window inference.
        """
        # 1. Acoustic Front-End VAD
        is_speech, energy, snr = self.vad.process_chunk(chunk)
        self.last_audio_energy = energy
        self.last_snr = snr
        self.is_speech_active = is_speech

        self.ring_buffer.push(chunk)

        # 2. VAD Silence Gating: Skip model inference in STANDBY during pure room silence
        if self.state == self.STATE_STANDBY and self.require_wake_word:
            buf_energy = self.vad.compute_energy(self.ring_buffer.get_window())
            if not is_speech and buf_energy < 0.0015:
                self.vad_silence_bypasses += 1
                return None

        # 3. Dynamic Acoustic Endpointing in LISTENING mode
        if self.state == self.STATE_LISTENING:
            ep_state = self.endpointer.process(len(chunk), is_speech)
            if ep_state == AdaptiveEndpointer.STATE_ENDPOINT_REACHED:
                self.endpointer.reset()
                return self._evaluate_and_dispatch_command(trigger="ENDPOINTING")

        return self.process_current_window()

    def classify_window(self, window_audio: np.ndarray) -> Tuple[str, float, float, Dict[str, float], np.ndarray]:
        """
        Runs neural network forward pass on a 1.5s audio array.
        Returns: (predicted_class, confidence, latency_ms, top3_dict, mel_spec_numpy)
        """
        t0 = time.perf_counter()

        # Dynamic gain normalization for quiet microphones
        peak = np.max(np.abs(window_audio))
        if 0.001 < peak < 0.40:
            scale = min(6.0, 0.45 / max(peak, 0.01))
            window_audio = window_audio * scale

        tensor_audio = torch.from_numpy(window_audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            mel_spec = self.frontend(tensor_audio)
            logits = self.model(mel_spec)
            probs = F.softmax(logits, dim=-1)[0]
            conf, pred_idx = torch.max(probs, dim=-1)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        self.latencies_ms.append(latency_ms)
        self.total_frames_processed += 1

        confidence = conf.item()
        class_idx = pred_idx.item()
        command_name = IDX_TO_CLASS[class_idx]

        # Top 3 predictions
        top_probs, top_indices = torch.topk(probs, k=min(3, len(probs)))
        top3 = {IDX_TO_CLASS[idx.item()]: float(p.item()) for p, idx in zip(top_probs, top_indices)}

        mel_numpy = mel_spec.squeeze().cpu().numpy()
        self.last_mel_spec = mel_numpy
        self.last_top3 = top3
        self.last_predicted_command = command_name
        self.last_confidence = confidence
        return command_name, confidence, latency_ms, top3, mel_numpy

    def process_current_window(self) -> Optional[Dict[str, Any]]:
        """
        Extracts current 1.5-second buffer and executes forward pass.
        Applies wake-word state machine, confidence filtering, and debouncing.
        """
        if not self.ring_buffer.is_filled:
            return None

        # Check for timeout first
        timeout_event = self.check_timeout()

        window_audio = self.ring_buffer.get_window()
        command_name, confidence, latency_ms, top3, mel_numpy = self.classify_window(window_audio)
        current_time = time.time()

        # Ignore noise and silence
        if command_name in ["_silence_", "_background_noise_"]:
            return timeout_event

        # Debounce filter
        time_since_last = current_time - self.last_detection_time
        if time_since_last < self.holdoff_seconds and command_name == self.last_detected_command:
            return timeout_event

        # -------------------------------------------------------------
        # STATE 1: STANDBY (Awaiting Wake Word "Hi" / "Hello")
        # -------------------------------------------------------------
        if self.state == self.STATE_STANDBY and self.require_wake_word:
            if command_name == WAKE_WORD_CLASS and confidence >= self.wake_confidence:
                self.state = self.STATE_LISTENING
                self.wake_time = current_time
                self.last_detection_time = current_time
                self.last_detected_command = command_name
                self.total_wakes_triggered += 1
                self.endpointer.reset()

                # Siri/Alexa instantaneous rising earcon
                VOICE_ENGINE.play_earcon("wake")

                # Stage 3: Compound One-Shot ("One-Breath") Utterance Pipeline:
                # Slices the trailing 1.0 second of audio in the buffer to check if user
                # spoke "Hi Dandan play music" in one continuous sentence.
                trailing_samples = int(SAMPLE_RATE * 1.0)
                trailing_audio = self.ring_buffer.get_recent(trailing_samples)
                trailing_energy = self.vad.compute_energy(trailing_audio)

                if trailing_energy >= self.vad.min_energy_threshold * 1.3:
                    padded_trailing = np.zeros(NUM_SAMPLES, dtype=np.float32)
                    padded_trailing[:len(trailing_audio)] = trailing_audio
                    t_cmd, t_conf, t_lat, t_top3, _ = self.classify_window(padded_trailing)

                    if t_cmd not in [WAKE_WORD_CLASS, "_silence_", "_background_noise_"] and t_conf >= self.confidence_threshold:
                        # Compound One-Shot execution!
                        self.compound_commands_triggered += 1
                        self.total_commands_triggered += 1
                        self.state = self.STATE_STANDBY
                        self.last_detected_command = t_cmd
                        action_msg = self.controller.execute_command(t_cmd, t_conf)
                        VOICE_ENGINE.play_earcon("success")
                        self.controller.unduck_media()

                        return {
                            "event": "COMPOUND_COMMAND_EXECUTED",
                            "wake_word": command_name,
                            "command": t_cmd,
                            "confidence": t_conf,
                            "latency_ms": latency_ms + t_lat,
                            "state": self.state,
                            "action_message": action_msg,
                            "top3": t_top3,
                            "timestamp": current_time,
                            "trigger": "COMPOUND_ONE_SHOT"
                        }

                action_msg = self.controller.trigger_wake()
                return {
                    "event": "WAKE_WORD_DETECTED",
                    "command": command_name,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "state": self.state,
                    "action_message": action_msg,
                    "top3": top3,
                    "timestamp": current_time,
                    "trigger": "SLIDING_WINDOW"
                }
            else:
                # Ignored while in standby
                return timeout_event

        # -------------------------------------------------------------
        # STATE 2: LISTENING (Awakened; awaiting command)
        # -------------------------------------------------------------
        elif self.state == self.STATE_LISTENING or not self.require_wake_word:
            # If user repeats wake word, refresh timeout
            if command_name == WAKE_WORD_CLASS and confidence >= self.wake_confidence:
                self.wake_time = current_time
                self.last_detection_time = current_time
                self.endpointer.reset()
                action_msg = self.controller.trigger_wake()
                VOICE_ENGINE.play_earcon("wake")
                return {
                    "event": "WAKE_REFRESHED",
                    "command": command_name,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "state": self.state,
                    "action_message": action_msg,
                    "top3": top3,
                    "timestamp": current_time
                }

            # Check command confidence
            if confidence < self.confidence_threshold:
                return timeout_event

            # VALID COMMAND DETECTED
            return self._evaluate_and_dispatch_command(
                trigger="SLIDING_WINDOW",
                pre_evaluated=(command_name, confidence, latency_ms, top3)
            )

        return None

    def _evaluate_and_dispatch_command(
        self,
        trigger: str = "SLIDING_WINDOW",
        pre_evaluated: Optional[Tuple[str, float, float, Dict[str, float]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates current audio buffer and dispatches recognized command.
        Called either by dynamic endpointing or sliding window evaluation.
        """
        if pre_evaluated:
            command_name, confidence, latency_ms, top3 = pre_evaluated
        else:
            window_audio = self.ring_buffer.get_window()
            command_name, confidence, latency_ms, top3, _ = self.classify_window(window_audio)

        current_time = time.time()

        if command_name in [WAKE_WORD_CLASS, "_silence_", "_background_noise_"]:
            return None

        if confidence < self.confidence_threshold:
            return None

        self.last_detection_time = current_time
        self.last_detected_command = command_name
        self.total_commands_triggered += 1

        action_msg = self.controller.execute_command(command_name, confidence)
        VOICE_ENGINE.play_earcon("success")

        if self.require_wake_word:
            self.state = self.STATE_STANDBY
            self.endpointer.reset()
            self.controller.unduck_media()

        return {
            "event": "COMMAND_EXECUTED",
            "command": command_name,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "state": self.state,
            "action_message": action_msg,
            "top3": top3,
            "timestamp": current_time,
            "trigger": trigger
        }

    def process_audio_clip(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> List[Dict[str, Any]]:
        """
        Feeds a complete audio clip through sliding windows (hop = 0.25s)
        and returns list of all triggered events.
        """
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            import torchaudio.transforms as T
            resampler = T.Resample(sample_rate, SAMPLE_RATE)
            audio_tensor = resampler(torch.from_numpy(audio_data).float())
            audio_data = audio_tensor.numpy()

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        hop_samples = int(SAMPLE_RATE * 0.25)  # 250 ms hop
        events = []

        # If clip is shorter than 1.5s, center/pad
        if len(audio_data) <= NUM_SAMPLES:
            padded = np.zeros(NUM_SAMPLES, dtype=np.float32)
            pad_start = (NUM_SAMPLES - len(audio_data)) // 2
            padded[pad_start:pad_start + len(audio_data)] = audio_data
            self.ring_buffer.clear()
            self.ring_buffer.push(padded)
            evt = self.process_current_window()
            if evt:
                events.append(evt)
        else:
            # Stream through sliding window
            self.ring_buffer.clear()
            for start in range(0, len(audio_data), hop_samples):
                chunk = audio_data[start:start + hop_samples]
                evt = self.feed_audio_chunk(chunk)
                if evt:
                    events.append(evt)

        return events

    def get_latency_profile(self) -> Dict[str, float]:
        """Returns latency statistics in milliseconds."""
        if not self.latencies_ms:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0}
        arr = np.array(self.latencies_ms)
        return {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "count": len(arr)
        }

