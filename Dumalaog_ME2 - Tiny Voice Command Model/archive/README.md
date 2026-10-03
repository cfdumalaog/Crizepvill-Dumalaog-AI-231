# Archived ME2 files

Nothing in this archive was permanently deleted. These items were moved out of
the active project paths to make the current two-model demo easier to find.

| Folder | Preserved content |
|---|---|
| `legacy_vcm/` | Superseded BC-ResNet/runtime prototype, its simulations and tests, legacy Raspberry Pi package, and old 26-class ZIP. |
| `legacy_vcm/root_launchers/` | Legacy root-level VCM launchers, batch scripts, and control center executable. |
| `legacy_vcm/desktop_recorder/` | Rejected native desktop Tk recorder application, builder, and launcher scripts (browser recorder `record_dataset.py` is the active recorder). |
| `model_candidates/classagreementvcm_candidate/` | Concluded Gold-dataset candidate model files, deployment JSON, install script, and candidate launchers (removed from Pi and unpromoted). |
| `experiments/classagreementvcm/` | Training and comparison scripts and candidate unit tests for the concluded Gold candidate. |
| `notebooks/` | Earlier model notebooks, the warm-started wake candidate report, and the SLURP comparison experiment. |
| `runs/` | Superseded training runs and checkpoints (including `me2-vcm-20261001-wake-refresh2` and old logs). |
| `legacy_vcm/scripts/` | Generator for the old SAPI notebook; its test is in `legacy_vcm/tests/`. |
| `history/` | Full conversation transcript retained for reference. |
| `planning/` | Historical project-plan workbook. |
| `scratch/` | One-off notebook-generation experiment. |
| `deployment_staging/` | Consolidated deployment staging runs and extracted Pi bundles; the last audited `TinyVCM_RPi5_Antigrav_RELEASE.zip` remains in `deployment/dist/`. |
| `notebooks/checkpoints/` | A distinct Jupyter checkpoint of the archived 26-class notebook. |
| `slides/` | Earlier benchmark and demo slide deck versions (Template v2, Demo 20260930, Demo 20261002, Gold Live Benchmark 20261003, Antigrav working drafts). |

The active intent ONNX remains `deployment/current_vcm/models/intent_int8.onnx`. The current
local binary wake ONNX is `deployment/current_vcm/models/binary_wake_int8.onnx`. Generated
deliverable PDFs are consolidated under `outputs/pdf/`. Raw datasets in `data/` were not moved or modified.
