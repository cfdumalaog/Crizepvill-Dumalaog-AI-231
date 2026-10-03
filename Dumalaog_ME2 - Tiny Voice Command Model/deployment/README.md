# ME2 - VCM on Raspberry Pi 5 deployment

`latest_release.json` identifies the newest prepared package and whether the Pi has been verified running it. `last_pi_deployment.json` contains the installation receipt, both live model hashes, and the ARM64 benchmark. The active source pair is `current_vcm/`; the Pi application is `/home/dalmacio/Desktop/dandan`.

## Isolated dataset comparison candidate

The scratch-trained comparison model is intentionally installed at `/home/dalmacio/Desktop/me2-dataset-candidate` and manually running on loopback port `7865`. It is a separate experiment; it did not replace or overwrite `dandan`. The wake model is retained byte-identically; the candidate intent gates are confidence `0.76` and margin `0.00`. GPIO and boot autostart are both disabled.

On Windows, double-click `Start-Dataset-Candidate.bat` to verify/start that candidate, establish the SSH tunnel, and open `http://127.0.0.1:7865/assistant`. Use `/studio` for diagnostics. Its Pi install/API/model hashes and CPU p95 are recorded in `classagreementvcm_candidate_deployment.json`. There is no microphone connected to the Pi, so the interface and inference pass are verified but live speech is not.

The candidate scored 78.95% accuracy / 79.12% macro F1 versus 79.64% / 79.47% for the current model on the single paired frozen test. Do not promote it based on the overall result; use the isolated live speech trial to decide whether its better real-voice source slices transfer to your microphone. Complete results and per-class scores are in `docs/evaluations/ME2_VCM_Dataset_Comparison_20261002.md` and `.pdf`.

From the project folder, synchronize the latest completed paired training run:

```powershell
& '..\..\.venv\Scripts\python.exe' -u scripts\deploy_latest_vcm.py
```

`scripts/train_me2_vcm.py` performs synchronization after training by default. `--train-only` explicitly skips it. A completed training run whose Pi sync fails exits with code 3 and records pending deployment. Incomplete runs cannot become the active pair.

The workflow checks class order and model hashes, replays the fixed 0.95 wake operating point, stages the pair, builds a versioned manual application ZIP, and copies it to `C:\Users\danda\Desktop\dandan`. It connects to `cfdfnjrpi5.local` using the existing SSH key and the previously trusted Pi host key, verifies every payload on ARM64, archives the prior app, then replaces only the user's Desktop `dandan` deployment. An app startup failure restores the previous folder. Neither Kiko folder nor `/home/dalmacio/vcm` is inspected or changed. Boot autostart stays disabled.

Double-click Desktop `Start-VCM.bat` to sync, connect, and open the Assistant through SSH at `http://127.0.0.1:7864/assistant`. `Update-VCM.bat` retries sync without opening a browser. A current verified pair is left running. An offline Pi cannot be updated; the prepared release stays pending until a later training sync or launcher retry succeeds. No background schedule is installed.

On the Pi, run `~/Desktop/dandan/launch-vcm.sh` manually. The microphone is automatically discovered and can be refreshed/selected live. The app binds only to `127.0.0.1:7860`; the PC tunnel is not public internet access.

The dataset is the ME2 Spoken Command Dataset. Per-class results and limitations are retained in `current_vcm/metadata.json`. Newest does not mean best: the human936 pair remains below the 95% goal for several classes.
