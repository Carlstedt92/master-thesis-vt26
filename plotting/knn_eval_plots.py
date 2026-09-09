"""Plot utilities for KNN evaluation curves."""

from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from datetime import datetime


def plot_roc_curve(y_true, y_score, output_path: str, model_name:str):
    """Create and save ROC curve plot."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"KNN (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (HIV) - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_precision_recall_curve(y_true, y_score, output_path: str, model_name:str):
    """Create and save Precision-Recall curve plot."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    baseline = float(np.mean(y_true))

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, linewidth=2, label=f"KNN (AP={pr_auc:.3f})")
    plt.hlines(
        y=baseline,
        xmin=0,
        xmax=1,
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline={baseline:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (HIV) - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def save_knn_eval_results(
    model_name: str,
    best_k: int,
    cv_roc_auc: float,
    cv_pr_auc: float,
    cv_f1: float,
    test_roc_auc: float,
    test_pr_auc: float,
    test_f1: float,
    test_balanced_acc: float,
    n_samples: int,
    output_path: str,
):
    """Save KNN evaluation results to JSON metadata file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model_name": model_name,
        "dataset": "HIV",
        "evaluation_method": "KNN (k-Nearest Neighbors) with distance weights",
        "cv_folds": 5,
        "best_k": best_k,
        "cv_metrics": {
            "roc_auc": round(cv_roc_auc, 4),
            "pr_auc": round(cv_pr_auc, 4),
            "f1": round(cv_f1, 4),
        },
        "test_metrics": {
            "roc_auc": round(test_roc_auc, 4),
            "pr_auc": round(test_pr_auc, 4),
            "f1": round(test_f1, 4),
            "balanced_accuracy": round(test_balanced_acc, 4),
        },
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved KNN evaluation results: {output}")
