"""Master training and export runner for TinyDSCNN-48 Antigrav on Option B dataset."""
import shutil
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tinyvcm_antigrav.training import prepare, train
from tinyvcm_antigrav.export import export_models


def main():
    print("=" * 70)
    print("  TINYDSCNN-48 ANTIGRAV: OPTION B TRAINING (100 HUMAN SPEAKERS)")
    print("=" * 70, flush=True)

    ctx = prepare(seed=231)
    results = train(ctx, epochs=35, batch_size=64, lr=0.003)

    val_x, _, _ = ctx['splits']['val']
    test_x, test_y, _ = ctx['splits']['test']

    print("\n" + "=" * 70)
    print("  EXPORTING STATIC INT8 ONNX & BENCHMARKING")
    print("=" * 70, flush=True)

    export_summary = export_models(results['run_dir'], val_x, test_x, test_y)

    # Copy models to project root models/ for easy deployment
    models_dest = ROOT / 'models'
    models_dest.mkdir(parents=True, exist_ok=True)

    src_int8 = results['run_dir'] / 'models' / 'antigrav_optionb_int8.onnx'
    src_fp32 = results['run_dir'] / 'models' / 'antigrav_optionb_fp32.onnx'

    if src_int8.exists():
        shutil.copy2(src_int8, models_dest / 'antigrav_optionb_int8.onnx')
        shutil.copy2(src_fp32, models_dest / 'antigrav_optionb_fp32.onnx')
        print(f"Copied exported ONNX models to {models_dest / 'antigrav_optionb_int8.onnx'}", flush=True)

    print("\n" + "=" * 70)
    print("  TRAINING & EXPORT COMPLETE!")
    print(f"  Best Val Accuracy:  {results['best_val_acc']*100:.2f}% (Epoch {results['best_epoch']})")
    print(f"  Test Accuracy FP32: {export_summary['test_accuracy_fp32_onnx']*100:.2f}%")
    print(f"  Test Accuracy INT8: {export_summary['test_accuracy_int8_onnx']*100:.2f}%")
    print(f"  Run Directory:      {results['run_dir']}")
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
