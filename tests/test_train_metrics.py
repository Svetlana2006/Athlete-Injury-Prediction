import numpy as np
import pytest

from app.ml.train import compute_metrics


def test_compute_metrics_reports_classification_ranking_and_top_k_scores():
    metrics = compute_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.4, 0.9]),
        top_k_pct=0.5,
    )

    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["top_k_precision"] == pytest.approx(0.5)
    assert metrics["top_k_recall"] == pytest.approx(0.5)
    assert metrics["top_k_lift"] == pytest.approx(1.0)
    assert np.isfinite(metrics["brier"])
    assert np.isfinite(metrics["log_loss"])
    assert np.isfinite(metrics["ece"])


def test_compute_metrics_marks_class_ranking_metrics_undefined_for_one_class():
    metrics = compute_metrics(np.array([0, 0]), np.array([0.1, 0.8]))

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["recall"] == 0.0
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 0)


@pytest.mark.parametrize(
    ("targets", "probabilities", "kwargs"),
    [
        (np.array([]), np.array([]), {}),
        (np.array([0, 1]), np.array([0.2]), {}),
        (np.array([0, 2]), np.array([0.2, 0.8]), {}),
        (np.array([0, 1]), np.array([0.2, 0.8]), {"top_k_pct": 0}),
        (np.array([0, 1]), np.array([0.2, 0.8]), {"decision_threshold": 1.1}),
        (np.array([0, 1]), np.array([0.2, np.nan]), {}),
    ],
)
def test_compute_metrics_rejects_invalid_inputs(targets, probabilities, kwargs):
    with pytest.raises(ValueError):
        compute_metrics(targets, probabilities, **kwargs)
