from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "ME1_MNIST_Manual_CNN.ipynb"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"


def test_notebook_exists_and_contains_required_sections() -> None:
    assert NOTEBOOK_PATH.exists()
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    all_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    required_phrases = (
        "Crizepvill Dumalaog",
        "three convolutional",
        "EPOCHS = 5",
        "Test accuracy",
        "predictions_4x4.png",
        "Ground truth",
        "Agent",
    )
    for phrase in required_phrases:
        assert phrase in all_source


def test_notebook_has_exactly_one_training_call_for_five_epochs() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert code.count("history = train_model(") == 1
    assert "EPOCHS = 5" in code
    assert "epochs=EPOCHS" in code


def test_executed_result_contract_when_metrics_exist() -> None:
    if not METRICS_PATH.exists():
        return
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    output_text = "\n".join(
        output.get("text", "")
        if isinstance(output.get("text", ""), str)
        else "".join(output.get("text", []))
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )

    assert metrics["epochs"] == 5
    assert len(metrics["history"]) == 5
    assert metrics["train_samples"] == 60_000
    assert metrics["test_samples"] == 10_000
    assert metrics["deterministic_algorithms"] is True
    assert 0.95 <= metrics["test_accuracy"] <= 1.0
    assert len(metrics["prediction_grid_indices"]) == 16
    assert f"Test accuracy: {metrics['test_accuracy'] * 100:.2f}%" in output_text
    assert (PROJECT_ROOT / "artifacts" / "predictions_4x4.png").exists()
