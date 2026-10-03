# Archived ME2 files

Nothing in this archive was permanently deleted. These items were moved out of
the active project paths to make the current two-model demo easier to find.

| Folder | Preserved content |
|---|---|
| `legacy_vcm/` | Superseded BC-ResNet/runtime prototype, its simulations and tests, legacy Raspberry Pi package, and old 26-class ZIP. |
| `notebooks/` | Earlier model notebooks, the warm-started wake candidate report, and the SLURP comparison experiment. |
| `runs/` | Superseded training runs/checkpoints. This path remains ignored by Git like the original `runs/` directory; files remain on disk. |
| `legacy_vcm/scripts/` | Generator for the old SAPI notebook; its test is in `legacy_vcm/tests/`. |
| `history/` | Full conversation transcript retained for reference. |
| `planning/` | Historical project-plan workbook. |
| `scratch/` | One-off notebook-generation experiment. |
| `deployment_staging/` | Superseded extracted Pi bundles and older ZIPs; the last audited `TinyVCM_RPi5_Antigrav_RELEASE.zip` remains in `deployment/dist/`. |
| `notebooks/checkpoints/` | A distinct Jupyter checkpoint of the archived 26-class notebook. |
| `slides/` | Intermediate working presentation drafts and earlier iterations of benchmark slide decks. |

The active intent ONNX remains `models/antigrav_optionb_int8.onnx`. The current
local binary wake ONNX is `models/binary_wake_int8.onnx`. The prior 32-class
Pi wake model remains `models/antigrav_wake32_int8.onnx` for the older verified
release/rollback path. Raw datasets in `data/` were not moved or modified.
