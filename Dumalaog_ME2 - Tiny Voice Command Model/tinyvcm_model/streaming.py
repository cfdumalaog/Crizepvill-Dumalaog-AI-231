"""Bounded speech endpoint detection shared by live audio and file replay.

The detector stores audio only in memory. A completed utterance is consumed once;
overlong speech is discarded until silence instead of classifying clipped words.
"""
from dataclasses import dataclass

import numpy as np

from .config import SR, SAMPLES


@dataclass
class SpeechFrame:
    rms: float
    voiced: bool
    started: bool = False
    completed: np.ndarray | None = None
    rejected: str | None = None


class UtteranceSegmenter:
    def __init__(self, threshold=.012, endpoint_seconds=.30, pre_roll_seconds=.15,
                 min_speech_seconds=.15, sample_rate=SR, max_samples=SAMPLES):
        if not 0 < threshold < 1:
            raise ValueError('Speech threshold must be between 0 and 1.')
        if endpoint_seconds < .15:
            raise ValueError('Endpoint silence must be at least 0.15 seconds.')
        self.threshold = float(threshold)
        self.endpoint_samples = round(endpoint_seconds * sample_rate)
        self.pre_roll_samples = round(pre_roll_seconds * sample_rate)
        self.min_speech_samples = round(min_speech_seconds * sample_rate)
        self.max_samples = max_samples
        self.reset()

    def reset(self):
        self.active = False
        self.discarding = False
        self.pre_roll = np.zeros(0, dtype=np.float32)
        self.parts = []
        self.total_samples = 0
        self.voiced_samples = 0
        self.silence_samples = 0

    def current_audio(self):
        return np.concatenate(self.parts) if self.parts else np.zeros(0, dtype=np.float32)

    def push(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if not len(chunk) or not np.all(np.isfinite(chunk)):
            self.reset()
            return SpeechFrame(0., False, rejected='invalid_audio')
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        frame = SpeechFrame(rms, rms > self.threshold)
        if self.discarding:
            self.silence_samples = 0 if frame.voiced else self.silence_samples + len(chunk)
            if self.silence_samples >= self.endpoint_samples:
                self.reset()
            return frame
        if not self.active:
            if not frame.voiced:
                self.pre_roll = np.concatenate((self.pre_roll, chunk))[-self.pre_roll_samples:]
                return frame
            self.active = True
            frame.started = True
            self.parts = [self.pre_roll.copy()]
            self.total_samples = len(self.pre_roll)

        self.parts.append(chunk.copy())
        self.total_samples += len(chunk)
        if frame.voiced:
            self.voiced_samples += len(chunk)
            self.silence_samples = 0
        else:
            self.silence_samples += len(chunk)

        # Ignore the endpoint's trailing silence when enforcing the model window.
        if self.total_samples - self.silence_samples > self.max_samples:
            self.reset()
            self.discarding = True
            frame.rejected = 'utterance_too_long'
        elif self.silence_samples >= self.endpoint_samples:
            audio = self.current_audio()
            # Keep one pre-roll-length tail, then center-pad in the model frontend.
            trim = max(0, self.silence_samples - self.pre_roll_samples)
            if trim:
                audio = audio[:-trim]
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            if self.voiced_samples >= self.min_speech_samples:
                frame.completed = audio
            else:
                frame.rejected = 'speech_too_short'
            self.reset()
        return frame
