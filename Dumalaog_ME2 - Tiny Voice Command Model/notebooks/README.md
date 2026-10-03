# ME2 training notebooks

Open **`ME2_Tiny_VCM_Training.ipynb`** for the current two-model project. It is executed with the shared `ai222-231` kernel and contains the data/label audit, architecture, training progress, per-class metrics, model hashes, and the frozen comparison between the current model and the new dataset-trained candidate. The new candidate is installed separately at `/home/dalmacio/Desktop/me2-dataset-candidate` on port 7865; the default `/home/dalmacio/Desktop/dandan` release remains unchanged. The notebook includes the candidate learning curve and all 31 per-class F1 comparisons.

By default the notebook reads saved runs and never retrains or deploys. Set `RETRAIN=True` only when you intend to start another scratch training run; that creates a versioned run. The published comparison test has already been scored once and must not be reused for selection or threshold tuning. The candidate Pi trial currently reports no connected microphone, so live speech testing remains open.

Older notebooks are retained under `../archive/notebooks/` for provenance. They document prior experiments and are not the instructions for the current demo.
