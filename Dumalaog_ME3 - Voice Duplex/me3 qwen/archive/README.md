# ME3 archive

This directory contains recoverable historical ME3 artifacts. Nothing in this
archive is the active runtime or the current presentation.

## Canonical active version

- Release bundle: `../dist/me3-20261103-172149-600e6b23/`
- Release ZIP: `../dist/me3-20261103-172149-600e6b23.zip`
- Presentation: `../docs/presentation/ME3_Voice_Duplex_Presentation_v9_Luna.pptx`
- ACM paper: `../docs/acm/ME3_Voice_Duplex_ACM.pdf` and its `.tex` source
- Active launcher set: `../launchers/`

The release manifest and workstation acceptance record remain in
`../deployment/`. The release is workstation-accepted and prepared; it is not
automatically a Pi deployment.

## What was archived on 2026-11-04 (Asia/Manila)

- `legacy-releases/dist/` contains the 46 older release directories and their
  matching ZIP files. These packages include historical copies of launcher
  files; their launcher copies are intentionally preserved inside the package
  that produced them.
- `legacy-presentations/presentation/` contains the older PowerPoint versions
  and presentation build/output directories. The Luna deck is the only active
  presentation left in `docs/presentation/`.
- `legacy-builds/root/` contains the old top-level LaTeX build directory.
- `legacy-builds/docs-acm/` contains the older ACM build directory. The current
  source and PDF remain directly under `docs/acm/`.

## What remains outside the archive

The seven files in `launchers/` are distinct documented entry points, not
duplicate versions: primary duplex voice, Qwen Instruct audio, diagnostic UI,
presenter web surface, Pi web tunnel, Pi injection demo, and class recorder.
They all remain active. Experiment logs under `runs/` and task artifacts under
`agent_workspace/` were not mass-moved because they are evidence or reusable
demo inputs rather than release versions. The protected `.env` was not moved
or opened.

To restore a historical artifact, copy or move it back only after checking its
manifest and verifying that the target path is not the active version.
