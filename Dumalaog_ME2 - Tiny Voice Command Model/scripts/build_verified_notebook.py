"""Build the current, explicitly provisional notebook; preserve legacy notebook."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('''# Tiny VCM — training from scratch and measured INT8 deployment
**Crizepvill F. Dumalaog · 202521406 · AI 231 ME2**

This is the current reproducible notebook. The older notebook is retained as historical work.
**Provisional dataset:** existing Windows SAPI synthetic speech is not recordings of three people.
No pretrained recognition weights or ASR are used. A fresh DS-CNN is initialized below.
Human recognition and Raspberry Pi performance are unverified until those experiments run.

Run **Run → Run All Cells**. Epoch output appears during training. Switch `DATA_MODE` to `human`
after completing `record_dataset.py` with three real speakers, every class and multiple conditions.
The last speaker ID is test-only; the penultimate is validation-only. Four or more people are preferable.
''')
code('''from pathlib import Path
import sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "tinyvcm").is_dir())
sys.path.insert(0, str(ROOT))
from tinyvcm.training import prepare, train, export_and_evaluate
DATA_MODE = "synthetic_baseline"  # change to "human" only after recording
EPOCHS = 100
ctx = prepare(DATA_MODE, seed=231)''')
md('''## Dataset and evaluation boundary
The synthetic baseline holds out one complete TTS voice. The remaining voices' +2 speech-rate
groups are validation only. Base utterances and their pitch/time variants stay together.
SHA-256 checks reject identical WAV files across splits. Synthetic noise is labeled as such.
Existing legacy audio is only 1.5 seconds and may already contain truncated phrases; this run cannot
repair missing speech. New human collection rejects clipped/overlong recordings.

There are 23 fixed intents spanning the earlier project's ten categories, one wake class and two
background classes. A collected human dataset adds unrelated speech (`_unknown_`). This recognizes
a limited vocabulary, not free-form conversation or arbitrary timer values.
''')
code('''import matplotlib.pyplot as plt
from tinyvcm.config import PHRASES
print("Labels:", len(ctx["labels"]))
for name in ctx["labels"]:
    print(f"{name:22} {PHRASES[name]}")
plt.figure(figsize=(10,3))
plt.imshow(ctx["splits"]["train"][0][0,0], origin="lower", aspect="auto")
plt.xlabel("10 ms frame"); plt.ylabel("Mel band"); plt.title("Shared train/runtime frontend: 40 × 151")
plt.show()''')
md('''## Acoustic frontend and network
At 16 kHz, a periodic 400-sample Hann window covers 25 ms, with a 160-sample (10 ms) hop
and a 512-point FFT. Reflect-centered framing gives 151 frames for 1.5 seconds.
Forty Slaney mel filters summarize squared FFT magnitude, followed by `log(power + 1e-6)`
and per-window standardization. Exactly the same NumPy implementation runs on Windows and Pi.

The DS-CNN uses a 48-channel stem, four depthwise/pointwise blocks, global average pooling and
a classifier. A depthwise 3×3 and pointwise 1×1 pair costs `9C + C²` weights instead of `9C²`.
This is a DS-CNN, not the published BC-ResNet-1; the old custom attention network remains in `src/`.
AdamW, train-only noise mixing and masking, and a cosine learning-rate schedule are used.
Validation selects the checkpoint. Test data is never used by the optimizer or INT8 calibration.
''')
code('''history = train(ctx, epochs=EPOCHS, batch_size=64)''')
code('''fig, ax = plt.subplots(1,2,figsize=(12,4))
for key in ["train_loss", "val_loss"]:
    ax[0].plot([r[key] for r in history], label=key)
for key in ["train_accuracy", "val_accuracy"]:
    ax[1].plot([r[key] for r in history], label=key)
for a in ax: a.legend(); a.set_xlabel("Epoch"); a.grid(alpha=.2)
ax[0].set_ylabel("Cross entropy"); ax[1].set_ylabel("Accuracy")
fig.tight_layout(); fig.savefig(ctx["run"] / "training_curves.png", dpi=140)
plt.show()''')
md('''## Static INT8 export and held-out testing
ONNX Runtime calibrates activations using training examples only. Every convolution and dense
weight tensor is checked for INT8 storage. This differs from dynamic quantization, which did not
quantize the earlier convolution layers. Frontend and output probabilities remain floating point.
The actual serialized ONNX must be less than **500,000 bytes**.

Metrics below are measured on this Windows PC, not on a Raspberry Pi. Noise curves use held-out
synthetic noise. The runtime's 1.5-second audio context and two-window confirmation add response
time beyond the neural inference measurement. No human false-alarm/hour claim is made.
''')
code('''metrics = export_and_evaluate(ctx)''')
code('''import numpy as np
fig, ax = plt.subplots(figsize=(11,10))
im=ax.imshow(metrics["confusion_matrix"],cmap="Blues")
ax.set_xticks(range(len(ctx["labels"])),ctx["labels"],rotation=90,fontsize=7)
ax.set_yticks(range(len(ctx["labels"])),ctx["labels"],fontsize=7)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("INT8 held-out synthetic voice — not human accuracy" if DATA_MODE != "human" else "INT8 held-out human speaker")
fig.colorbar(im,ax=ax); fig.tight_layout()
fig.savefig(ctx["run"] / "confusion_matrix.png",dpi=140); plt.show()
print("Run artifacts:",ctx["run"])
print("PC: Start-PC-Demo.ps1 at workspace root. Say Hi Dandan, pause for LISTENING, then a short command.")''')
md('''## What remains for assignment completion
1. Record consenting humans (at least three), unrelated speech and real room noise under multiple
conditions. Retrain with `DATA_MODE="human"`. Synthetic voices do not meet this requirement.
2. Validate wake false rejects and false alarms over timed real continuous audio. Tune thresholds
using validation speakers only; report held-out results once.
3. Flash **Raspberry Pi OS 64-bit**, transfer the generated `release/` folder and run `setup_rpi.sh`.
The installer downloads dependencies once; runtime then works offline. A folder of Python files
is not a bootable SD image. Use USB or HDMI audio on Pi 5 (no built-in 3.5 mm audio jack).
4. Run `python -m tinyvcm.runtime --output pi_benchmark.json` on the actual Pi. Verify p95 inference,
resident memory, microphone capture, and wired LED/buzzer before declaring deployment complete.

References: [class lessons](https://github.com/roatienza/Deep-Learning-Experiments),
[PyTorch quantization](https://docs.pytorch.org/tutorials/recipes/quantization.html),
[Pi OS setup](https://www.raspberrypi.com/documentation/computers/getting-started.html).
Agent contribution: Codex audited earlier claims and generated this pipeline. Student review is still required.
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (AI 222 + AI 231)','language':'python','name':'ai222-231'}})
path=root/'notebooks/ME2_From_Scratch_Verified.ipynb'
nbf.write(nb,path)
print(path)
