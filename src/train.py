"""
train.py — Training Loop with Early Stopping and Gradient Clipping
====================================================================

This module implements the training procedure for the T3C model:

    1. Balanced class weights via sklearn's compute_class_weight
    2. Weighted CrossEntropyLoss to handle ASHRAE II class imbalance
    3. Adam optimizer with ReduceLROnPlateau scheduler
    4. Gradient clipping (max_norm=1.0) to stabilize Transformer training
    5. Early stopping on validation loss to prevent overfitting
    6. Separate train / val accuracy tracking per epoch

Key Improvement over Original Notebook:
    The original notebook used the test set as the validation set during
    training (for LR scheduling). This module introduces a proper 3-way
    split (train/val/test) so that:
      - Validation set guides hyperparameter decisions (scheduler, early stop)
      - Test set is truly held out and only used for final evaluation

Reference: Paper Section 3.2 (Experimental Setup).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append("..")
from config import TrainConfig, ModelConfig
from .model import ComfortTransformer


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility.

    Ensures identical results across runs by controlling randomness in
    NumPy, PyTorch CPU, PyTorch CUDA, and cuDNN backends.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    X_train_seq: np.ndarray,
    y_train_enc: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_enc: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convert NumPy arrays to PyTorch DataLoaders.

    Parameters
    ----------
    X_train_seq : np.ndarray
        Training sequences from PISSG, shape (N_train, seq_length, features).
    y_train_enc : np.ndarray
        Label-encoded training targets.
    X_val_seq : np.ndarray
        Validation sequences from PISSG.
    y_val_enc : np.ndarray
        Label-encoded validation targets.
    batch_size : int
        Batch size for DataLoaders.

    Returns
    -------
    train_loader, val_loader : tuple of DataLoader
    """
    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_enc, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_enc, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(
    X_train_seq: np.ndarray,
    y_train_enc: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_enc: np.ndarray,
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[ComfortTransformer, Dict[str, List[float]]]:
    """
    Train the T3C model with early stopping.

    Parameters
    ----------
    X_train_seq : np.ndarray
        PISSG-generated training sequences, shape (N, T, D).
    y_train_enc : np.ndarray
        Label-encoded training targets.
    X_val_seq : np.ndarray
        PISSG-generated validation sequences.
    y_val_enc : np.ndarray
        Label-encoded validation targets.
    model_cfg : ModelConfig, optional
        Model architecture config.
    train_cfg : TrainConfig, optional
        Training hyperparameters.
    device : torch.device, optional
        Compute device (auto-detected if not provided).

    Returns
    -------
    model : ComfortTransformer
        Trained model (best checkpoint by validation loss).
    history : dict
        Training history with keys: 'train_loss', 'val_loss',
        'train_acc', 'val_acc'.
    """
    if model_cfg is None:
        model_cfg = ModelConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(train_cfg.random_seed)

    # ------------------------------------------------------------------
    # 1. Build DataLoaders
    # ------------------------------------------------------------------
    train_loader, val_loader = build_dataloaders(
        X_train_seq, y_train_enc, X_val_seq, y_val_enc,
        batch_size=train_cfg.batch_size,
    )

    # ------------------------------------------------------------------
    # 2. Initialize model
    # ------------------------------------------------------------------
    model = ComfortTransformer(
        model_cfg=model_cfg,
        jitter_std=train_cfg.jitter_std,
    ).to(device)

    print(f"Model initialized on: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------
    # 3. Loss function with balanced class weights
    # ------------------------------------------------------------------
    # The ASHRAE II dataset has significant class imbalance (Neutral is
    # the majority class). Balanced weights ensure the model doesn't
    # simply predict "Neutral" for everything.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_enc),
        y=y_train_enc,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # ------------------------------------------------------------------
    # 4. Optimizer and scheduler
    # ------------------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_cfg.scheduler_patience,
        factor=train_cfg.scheduler_factor,
    )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\nStarting training for {train_cfg.epochs} epochs...")
    print("-" * 75)

    for epoch in range(train_cfg.epochs):

        # --- Phase 1: Training ---
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, augment=True)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping: prevents exploding gradients in Transformers
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg.max_grad_norm
            )

            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)

        # --- Phase 2: Validation ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        train_correct = 0
        train_total = 0

        with torch.no_grad():
            # Validation metrics
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X, augment=False)

                v_loss = criterion(outputs, batch_y)
                val_running_loss += v_loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

            # Training accuracy (separate pass without augmentation)
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X, augment=False)
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

        # --- Phase 3: Metrics ---
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        train_acc = train_correct / train_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # --- Phase 4: Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = " * (best)"
        else:
            patience_counter += 1
            marker = ""

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1:3d}/{train_cfg.epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | "
            f"LR: {current_lr:.2e}{marker}"
        )

        if patience_counter >= train_cfg.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {train_cfg.early_stop_patience} epochs)")
            break

    # ------------------------------------------------------------------
    # 6. Restore best model
    # ------------------------------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (val_loss={best_val_loss:.4f})")

    return model, history
