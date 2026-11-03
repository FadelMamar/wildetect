"""Classification metrics shared by training and evaluation."""

from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    Precision,
    Recall,
)


def build_classification_metrics(num_classes: int) -> dict:
    """Per-class multiclass metrics.

    Both the Lightning module and ``ClassificationEvaluator`` must call this
    so a score from training and a score from evaluation are the same function
    of the same probabilities.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    config = {
        "task": "multiclass",
        "num_classes": num_classes,
        "average": None,
    }
    return {
        "accuracy": Accuracy(**config),
        "precision": Precision(**config),
        "recall": Recall(**config),
        "f1": F1Score(**config),
        "auroc": AUROC(**config),
        "average_precision": AveragePrecision(**config),
    }
