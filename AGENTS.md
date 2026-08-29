# AI 231 repository instructions

These instructions apply to this repository and all future AI 231 work.

## Before working

1. Read `README.md` and the README for the target exercise.
2. If this checkout is inside the shared `AI 222 231` workspace, also read the
   parent `AGENTS.md`, `HANDOFF_INDEX.md`, and `SETUP_NOTES.md` completely.
3. Inspect Git status and preserve unrelated/user changes.
4. Use the shared parent environment at `..\.venv`; do not create another
   environment in this repository.

## Repository organization

- Keep one directory per machine exercise or project.
- Keep this repository as the single GitHub repository for all AI 231 work.
- Update the root exercise table and the target exercise README after material
  changes.
- Commit source, executed submission notebooks, compact metrics, and report
  figures. Do not commit downloaded datasets, caches, or model checkpoints.

## ME1 restrictions

Work under `ME1_CNN_Einops` must preserve the assignment boundary:

- Three convolutional stages implemented manually.
- Convolution uses tensor window views, Einops, and `einsum`.
- Pooling uses Einops reduction; dense transformations use `einsum`.
- No PyTorch neural-network layer API, functional CNN operation, pretrained
  model, TensorFlow/Keras CNN, or NLP library.
- Torchvision is permitted only as the official MNIST dataset reader.
- The submission notebook must retain the executed five-epoch evidence, test
  accuracy, and 16-image 4×4 GT/prediction grid.

Run `python -m pytest -q` from `ME1_CNN_Einops` after changing ME1.

## Security and handoff

- Never commit API keys, credentials, tokens, VPN files, or secret fragments.
- If the parent workspace exists, refresh its handoff with
  `..\scripts\Update-HandoffIndex.ps1`, then update the substantive handoff and
  change log before completing material work.
- In a standalone clone, record current results, verification, and remaining
  work in the relevant exercise README and the root README before handoff.
