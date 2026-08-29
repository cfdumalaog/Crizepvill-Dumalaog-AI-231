# PyTorch Beginner Manual and Iris Project

This folder contains a beginner-friendly PDF manual plus a complete neural
network example for classifying Iris flowers.

## Run the example

Activate the one shared environment from the `AI 222 231` root, then run the
script:

```powershell
Set-Location 'C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231'
& '.\.venv\Scripts\Activate.ps1'
python '.\AI 231\pytorch_iris_example.py'
```

The script creates or refreshes:

- `iris_sample.csv` - 150 labeled flower records
- `output/pytorch_iris_example/iris_mlp_checkpoint.pt` - saved model weights
- `output/pytorch_iris_example/metrics.json` - reproducible run summary
- three PNG figures used in the manual

Open `output/pdf/pytorch_beginner_manual.pdf` for the full guided tutorial.

The included model is an educational example, not a production benchmark.
