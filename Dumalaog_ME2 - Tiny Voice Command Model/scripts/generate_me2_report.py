"""Write the evidence-based LaTeX technical report from frozen run artifacts."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DOCS = PROJECT / 'docs'
STAGED = PROJECT / 'deployment/current_vcm'


def tex(value) -> str:
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def main() -> None:
    meta = json.loads((STAGED / 'metadata.json').read_text(encoding='utf-8'))
    data = json.loads((DOCS / 'ME2_DATASET_ALIGNMENT.json').read_text(encoding='utf-8'))
    arch = json.loads((DOCS / 'ME2_ARCHITECTURE_VERIFIED.json').read_text(encoding='utf-8'))
    point = meta['wake']['operating_point_replay']
    intent = meta['intent']
    personal_report = intent['personal_heldout']['classification_report']
    per_class = intent['dataset_test_per_label']
    below = [label for label, row in per_class.items() if row['f1'] < 0.95 or row['recall'] < 0.95]
    stage_rows = '\n'.join(
        f"{tex(name)} & {tex(shape)} & {params:,} \\\\"
        for name, shape, params in arch['intent_stages']
    )
    class_rows = '\n'.join(
        f"{tex(label)} & {row['f1']*100:.1f} & {row['recall']*100:.1f} & {row['support']} & "
        f"{int(personal_report.get(label, {}).get('support', 0))} \\\\"
        for label, row in per_class.items()
    )
    group_rows = '\n'.join(
        f"{tex(name)} & {tex(', '.join(labels))} \\\\"
        for name, labels in data['course_command_groups'].items()
    )
    content = r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,longtable,graphicx,xcolor,hyperref,array}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\setlength{\parskip}{6pt}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}
\begin{document}
\begin{center}
{\LARGE\bfseries ME2 - VCM on Raspberry Pi 5\\[5pt]}
{\large Two-stage, from-scratch voice command model}\par
Crizepvill F. Dumalaog \quad AI 231 Machine Exercise 2 \quad 30 September 2026
\end{center}

\begin{abstract}
This report documents the current training artifacts and code, including measured limits. A binary 2-class wake detector continuously evaluates local audio, then a 31-class intent classifier maps an accepted command to a coded action. Both use the same TinyDSCNN-48 topology and a 40-band log-mel frontend. Weather uses a separate live API; recognition itself uses no cloud model. The staged INT8 pair totals \textbf{@@MODEL_BYTES@@ bytes}. The 95\% per-class goal is assessed on a reused speaker-held-out command set and is not assumed to be met.
\end{abstract}

\section{Requirements and command mapping}
The course asks for common smart-device interactions, multi-speaker speech, scratch training, a tiny offline classifier, and a Raspberry Pi demonstration. The class dataset contains @@DATASET_ROWS@@ command WAV rows from @@DATASET_SPEAKERS@@ reference speaker IDs. The 31 output labels refine the ten requested command families into fixed slot values. The wake phrase is a separate binary output, not a 32nd intent class.

\begin{longtable}{p{0.28\textwidth}p{0.65\textwidth}}
\toprule Course command family & Dataset labels \\
\midrule\endhead
@@GROUP_ROWS@@
\bottomrule
\end{longtable}

The recorder's 31 intent choices equal the model label order and manifest label set. Each fixed suggested phrase appears in the matching class's training transcripts, so the suggested prompts were retained rather than changed after collection. Four additional recorder choices are \texttt{wake\_word}, \texttt{\_unknown\_}, \texttt{\_background\_noise\_}, and \texttt{\_silence\_}. Exact legacy command mappings are accepted; unsupported old slot values stay excluded. The machine-readable check is \texttt{docs/ME2\_DATASET\_ALIGNMENT.json}.

\section{Data provenance and split discipline}
The dataset's local README attributes its reference voices to LibriSpeech and SilencioPH. The manifest uses @@TRAIN_ROWS@@ train, @@VAL_ROWS@@ validation, and @@TEST_ROWS@@ test rows, with reference speakers separated by partition. WAVs are read and resampled to 16 kHz by the frontend when necessary. The human recorder holds @@HUMAN_ROWS@@ rows from three anonymous IDs (@@SPEAKER_ROWS@@); exact waveform deduplication reduces them to 292 unique waveforms. Only @@MAPPED_CLASSES@@ of 31 classes have mapped personal examples; @@MISSING_CLASSES@@ lack any. The personal split is by deduplicated waveform within label, so the same speaker can occur in training and test. These personal scores are \emph{not} unseen-speaker evidence.

The wake optimizer labels every command in the reference \emph{training} partition as \texttt{NON\_WAKE}, along with training-only personal commands, unrelated speech, room noise and silence. The positive class is the recorded wake phrase. Validation and test partitions are not optimization inputs. Class weighting counters the large negative-to-positive imbalance. The intent optimizer uses the 31-class reference training partition and exact-mapped personal training takes. No pretrained weights or ASR model are loaded.

\section{Signal path and architecture}
Capture is mono at 16 kHz. A 2.5 s window contains 40,000 samples. The frontend computes a 512-point short-time Fourier transform using a 400-sample Hann window and 160-sample hop, projects power to 40 mel bands, applies log compression, then standardizes the feature tensor. The resulting network input is $(1,40,251)$ per item. The 25 ms window, 10 ms hop, and FFT size are distinct quantities; the 512-point transform zero-pads each 400-sample analysis frame.

\begin{figure}[htbp]\centering
\includegraphics[height=0.70\textheight,keepaspectratio]{ME2_two_stage_connections.pdf}
\caption{The actual two-model routing. The wake gate precedes intent inference; actions are coded, not generated by an LLM.}
\end{figure}

The stem is a $5\times5$ convolution with stride $(2,2)$, 48 output channels, batch normalization and ReLU. Four depthwise-separable blocks each contain a $3\times3$ channel-wise convolution, batch normalization, ReLU, a $1\times1$ pointwise convolution, batch normalization and ReLU. Block 2 halves the time dimension. Global average pooling yields 48 features, followed by dropout $p=0.15$ during training and a linear head. Only that head differs between the two models.

\begin{longtable}{p{0.48\textwidth}p{0.28\textwidth}r}
\toprule Stage & Output $(C,F,T)$ or vector & Parameters \\
\midrule\endhead
@@STAGE_ROWS@@
\midrule \textbf{Intent total} & \textbf{31 logits} & \textbf{@@INTENT_PARAMS@@} \\
\textbf{Wake total} & \textbf{2 logits} & \textbf{@@WAKE_PARAMS@@} \\
\bottomrule\end{longtable}

\begin{figure}[htbp]\centering
\includegraphics[height=0.72\textheight,keepaspectratio]{ME2_TinyDSCNN48_connections.pdf}
\caption{Layer-stage connections drawn with NetworkX from a live PyTorch shape and parameter inspection.}
\end{figure}

For 48 input and output channels, a conventional $3\times3$ convolution has $9\cdot48^2=20,736$ kernel weights. Each depthwise/pointwise pair has $9\cdot48+48^2=2,736$ kernel weights, before normalization parameters: about $7.58$ times fewer convolution weights. The full heads contain @@WAKE_PARAMS@@ and @@INTENT_PARAMS@@ parameters. Static QDQ INT8 ONNX export uses training-only calibration features.

\section{Training and measurements}
Both models were randomly initialized and trained in the shared PyTorch environment. The wake model used seed 231 for 30 epochs. The selected refined intent model used seed 232 for 60 epochs, 0.10 dropout, weighted cross-entropy without label smoothing, and a validation score that also considers the weakest reference class. AdamW with cosine learning-rate decay optimized both models; time/frequency masks and limited temporal jitter augmented training. Checkpoints and wake calibration used validation data. The reference test partition was used in earlier project iterations, so its final score is a reused benchmark rather than a pristine one-time test. The exact selected run is \texttt{@@RUN_NAME@@}.

At the requested fixed operating threshold of 0.95, the saved-clip wake replay accepted \textbf{@@WAKE_HITS@@/@@WAKE_TOTAL@@} personal wake clips, falsely accepted \textbf{@@WAKE_PERSONAL_FP@@/@@WAKE_PERSONAL_NEG@@} personal non-wake clips, and falsely accepted \textbf{@@WAKE_DATA_FP@@/@@WAKE_DATA_NEG@@} reference command clips. Binary F1 is @@WAKE_F1@@\% and binary accuracy is @@WAKE_ACC@@\% on this heavily imbalanced finite set; balanced accuracy is @@WAKE_BALACC@@\%. Five temporal positions per personal WAV are collapsed to one decision per clip. These counts do not measure live microphone recall or false wake activations per hour. The validation-selected threshold is @@VAL_THRESHOLD@@; 0.95 is a separately fixed demo operating choice.

The INT8 intent classifier reached \textbf{@@INTENT_ACC@@\% accuracy} and \textbf{@@INTENT_F1@@\% macro F1} on @@TEST_ROWS@@ reference command test clips. On @@PERSONAL_TEST@@ mapped personal held-out clips it reached @@PERSONAL_ACC@@\% accuracy and @@PERSONAL_F1@@\% macro F1, with overlapping speaker IDs. The weakest reference class F1 is @@MIN_F1@@\%. @@BELOW_COUNT@@ classes fall below 95\% on F1 or recall: @@BELOW_LABELS@@. A goal of at least 95\% on every class therefore remains unmet unless this list is empty. A per-class 95\% claim also needs substantially more unseen-speaker personal recordings.

\small
\begin{longtable}{p{0.47\textwidth}rrrr}
\toprule Intent class & F1 (\%) & Recall (\%) & Ref. test & Personal test \\
\midrule\endhead
@@CLASS_ROWS@@
\bottomrule\end{longtable}
\normalsize

\section{Runtime, controls, and hardware}
The standby state runs only the binary wake model after a simple energy gate. Once awake, the system waits for an utterance endpoint and runs the intent model. The user has ten seconds of inactivity before standby. The current Windows Studio shows a large RGB light simulation, numeric brightness, a numeric volume bar, thermostat target, and command log. The device module implements fixed light levels/colors, playback/next/pause/stop, volume steps, timer/alarm state, and temperature setpoints. A local generated melody provides deterministic music for an offline demo. Calls and messages are explicitly simulations; no telephony API is connected.

Weather is fetched asynchronously from the Open-Meteo current-weather API when networking is available; no weather API key is required. A failure returns a clear unavailable message without blocking microphone inference. Recognition remains local. The UI can change its capture microphone without reloading models; it retries after disconnects and falls back to an available input. Browser audio output selection uses \texttt{setSinkId} where supported; browser speech synthesis follows the browser's default output. A separate Restart VCM control remains for full service recovery. The recorder now exposes an immediate \textbf{Record again} control after saving.

\section{Raspberry Pi deployment and remaining gates}
The candidate bundle is built by \texttt{deployment/package\_personalized\_rpi.py}. It is a versioned application ZIP, not an SD-card image. It packages both INT8 ONNX models, their SHA-256 metadata, the same frontend and action code, local demo music, a manual \texttt{start-vcm.sh} launcher, and an on-device verifier. No systemd autostart is installed. The app binds to Pi loopback and can be viewed over an SSH tunnel. The earlier Pi release measured frontend-plus-model $p_{95}$ of 5.529 ms for wake and 5.270 ms for intent on ARM64, but those measurements are \emph{not} evidence for this newly trained pair or the latest UI code. The Pi was offline during this update, so live USB-microphone speech, device switching, sound output, latency and GPIO must be retested there.

For a class demonstration, show the wake transition first, then light brightness/color, volume, local music, temperature, and weather. State plainly that the RGB panel and thermostat are simulations unless GPIO/HVAC hardware is connected. For a stronger performance claim, record multiple sessions and all 31 classes from additional speakers, hold complete speakers out, collect room noise and near-miss wake phrases, and measure continuous false activations per hour. Create a fresh sealed test set for the next evaluation.

\end{document}
'''
    replacements = {
        '@@MODEL_BYTES@@': sum(meta[k]['model']['size_bytes'] for k in ('wake', 'intent')),
        '@@DATASET_ROWS@@': data['dataset_rows'], '@@DATASET_SPEAKERS@@': data['dataset_speakers'],
        '@@TRAIN_ROWS@@': data['dataset_split_counts']['train'],
        '@@VAL_ROWS@@': data['dataset_split_counts']['val'],
        '@@TEST_ROWS@@': data['dataset_split_counts']['test'],
        '@@GROUP_ROWS@@': group_rows, '@@STAGE_ROWS@@': stage_rows, '@@CLASS_ROWS@@': class_rows,
        '@@HUMAN_ROWS@@': data['human_manifest_rows'],
        '@@SPEAKER_ROWS@@': ', '.join(f'{k}: {v}' for k, v in data['human_speaker_rows'].items()),
        '@@MAPPED_CLASSES@@': len(data['human_mapped_command_rows']),
        '@@MISSING_CLASSES@@': len(data['intent_classes_with_no_personal_rows']),
        '@@INTENT_PARAMS@@': f"{arch['intent_parameters']:,}", '@@WAKE_PARAMS@@': f"{arch['wake_parameters']:,}",
        '@@RUN_NAME@@': tex(meta['source_run']),
        '@@WAKE_HITS@@': point['personal_wake_hits'], '@@WAKE_TOTAL@@': point['personal_wake_support'],
        '@@WAKE_PERSONAL_FP@@': point['personal_nonwake_false_accepts'],
        '@@WAKE_PERSONAL_NEG@@': point['personal_nonwake_support'],
        '@@WAKE_DATA_FP@@': point['dataset_command_false_accepts'],
        '@@WAKE_DATA_NEG@@': point['dataset_command_support'],
        '@@WAKE_F1@@': f"{point['binary_f1']*100:.1f}",
        '@@WAKE_ACC@@': f"{point['binary_accuracy']*100:.1f}",
        '@@WAKE_BALACC@@': f"{point['balanced_accuracy']*100:.1f}",
        '@@VAL_THRESHOLD@@': f"{meta['wake']['validation_selected_threshold']:.4f}",
        '@@INTENT_ACC@@': f"{intent['dataset_test_accuracy']*100:.1f}",
        '@@INTENT_F1@@': f"{intent['dataset_test_macro_f1']*100:.1f}",
        '@@PERSONAL_TEST@@': intent['personal_heldout']['count'],
        '@@PERSONAL_ACC@@': f"{intent['personal_heldout']['accuracy']*100:.1f}",
        '@@PERSONAL_F1@@': f"{intent['personal_heldout']['macro_f1_supported_classes']*100:.1f}",
        '@@MIN_F1@@': f"{min(row['f1'] for row in per_class.values())*100:.1f}",
        '@@BELOW_COUNT@@': len(below), '@@BELOW_LABELS@@': tex(', '.join(below)) if below else 'none',
    }
    for key, value in replacements.items():
        content = content.replace(key, str(value))
    out = DOCS / 'ME2_VCM_Technical_Report.tex'
    out.write_text(content, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
