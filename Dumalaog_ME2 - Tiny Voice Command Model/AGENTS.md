# ME2 VCM workflow

Follow the parent workspace and AI 231 instructions and preserve both Kiko folders.

- The active application is `vcm_app.py`; training/frontend code is `tinyvcm_model/`.
- Use the single shared workspace `.venv` for Windows training and deployment.
- After a completed paired training run, synchronize it to the Pi using
  `scripts/deploy_latest_vcm.py`. `scripts/train_me2_vcm.py` already does this
  by default. For other training paths, run synchronization explicitly.
- The Pi stays at `/home/dalmacio/Desktop/dandan`, and the Windows package and
  launcher stay at `C:\Users\danda\Desktop\dandan`. Do not enable boot autostart.
- Read `deployment/latest_release.json` and `deployment/last_pi_deployment.json`
  to distinguish prepared, pending, and verified deployed states. A successful
  local training/export is not proof of Pi deployment.
- If the Pi is offline, preserve the prepared package and report pending sync.
  Retry with Desktop `Update-VCM.bat` or `Start-VCM.bat` when it reconnects.
- Compare the running API's source run and both model hashes with the newest
  completed pair. Never infer live model identity solely from files on disk.
- Keep old releases only as rollback/archive evidence, separate from the active
  pair. Do not call the newest model the best model without measured evidence.
