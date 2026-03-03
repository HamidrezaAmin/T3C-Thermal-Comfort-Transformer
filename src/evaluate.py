"""
evaluate.py — Model Evaluation, Metrics, and Visualization
============================================================

This module provides:
    1. Test-set evaluation (loss, accuracy)
    2. Confusion matrix visualization
    3. Per-class classification report (precision, recall, F1)
    4. Training history plots (loss and accuracy curves)

The confusion matrix and F1 scores are critical for this project because
the ASHRAE II dataset has significant class imbalance. Overall accuracy
alone can be misleading — a model that always predicts "Neutral" would
achieve ~45% accuracy but be useless for detecting discomfort states.

Reference: Paper Section 4 (Results), Section 5.2 (Class-Wise Analysis).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)
from typing import Dict, List, Optional, Tuple

from .model import ComfortTransformer


def evaluate_model(
    model: ComfortTransformer,
    X_test_seq: np.ndarray,
    y_test_enc: np.ndarray,
    class_names: List[str],
    criterion: Optional[nn.Module] = None,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate the trained model on the held-out test set.

    Generates:
        - Test loss and accuracy
        - Confusion matrix heatmap
        - Per-class precision, recall, and F1 scores
        - Macro-averaged F1 (the primary metric for imbalanced classes)

    Parameters
    ----------
    model : ComfortTransformer
        Trained T3C model.
    X_test_seq : np.ndarray
        PISSG-generated test sequences, shape (N, T, D).
    y_test_enc : np.ndarray
        Label-encoded test targets.
    class_names : list of str
        Human-readable class names, e.g., ['Neutral', 'Uncomfortably Cool', 'Uncomfortably Warm'].
    criterion : nn.Module, optional
        Loss function. If None, uses unweighted CrossEntropyLoss.
    batch_size : int
        Batch size for evaluation DataLoader.
    device : torch.device, optional
        Compute device. Auto-detected if not provided.
    save_path : str, optional
        If provided, saves the confusion matrix figure to this path.

    Returns
    -------
    metrics : dict
        Dictionary containing 'test_loss', 'test_accuracy', 'macro_f1',
        and per-class F1 scores.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Build DataLoader
    test_dataset = TensorDataset(
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(y_test_enc, dtype=torch.long),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Inference ---
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X, augment=False)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Metrics ---
    test_loss /= len(test_loader)
    test_acc = np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2%}")
    print(f"  Macro F1:      {macro_f1:.4f}")
    print()

    # --- Classification Report ---
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
    )
    print(report)

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, class_names, save_path)

    # Compile metrics dict
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
    }
    for i, name in enumerate(class_names):
        metrics[f"f1_{name}"] = per_class_f1[i]

    return metrics


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix from sklearn.metrics.confusion_matrix.
    class_names : list of str
        Labels for each class.
    save_path : str, optional
        Path to save the figure. Displays interactively if None.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Thermal Comfort Confusion Matrix", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : dict
        Training history from train_model(), containing keys:
        'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    save_path : str, optional
        Path to save the figure. Displays interactively if None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    axes[0].plot(history["train_acc"], label="Training Accuracy")
    axes[0].plot(history["val_acc"], label="Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("(a) Model Accuracy", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Loss ---
    axes[1].plot(history["train_loss"], label="Training Loss")
    axes[1].plot(history["val_loss"], label="Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("(b) Model Loss", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)
