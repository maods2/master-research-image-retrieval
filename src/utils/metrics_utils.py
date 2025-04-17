from typing import Dict, Any, Callable, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, logger: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Compute classification metrics from true labels and predictions.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        logger: Optional logging function.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    conf = confusion_matrix(y_true, y_pred)
    
    if logger:
        logger(f"[Eval] Accuracy:  {acc:.4f}")
        logger(f"[Eval] Precision: {prec:.4f}")
        logger(f"[Eval] Recall:    {rec:.4f}")
        logger(f"[Eval] F1-Score:  {f1:.4f}")
        logger(f"[Eval] Confusion Matrix:\n{conf}")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": conf.tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }