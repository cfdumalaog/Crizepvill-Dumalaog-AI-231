# ME2 - VCM on Raspberry Pi 5: current two-model candidate

Wake trained from random initialization in `me2-vcm-20261001-wake-refresh`; intent weights retained byte-identical from `me2-vcm-20261001-human936`. Separate binary wake and 31-class intent INT8 ONNX models are staged here with exact class ordering and SHA-256 hashes. The local operating wake threshold is fixed at 0.95. See `metadata.json` for held-out scores and limitations. This directory is the source for the next manual-launch Pi bundle; its presence alone does not mean the Pi has been updated.
