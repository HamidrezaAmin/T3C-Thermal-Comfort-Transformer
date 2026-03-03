"""
run_train.py — End-to-End Training Pipeline
=============================================

Usage:
    python scripts/run_train.py --data_path data/df1.csv
    python scripts/run_train.py --data_path data/df1.csv --epochs 100 --batch_size 128
    python scripts/run_train.py --data_path data/df1.csv --d_model 128 --nhead 4

This script runs the complete T3C pipeline:
    1. Load and preprocess the ASHRAE II dataset
    2. Generate temporal sequences via PISSG
    3. Train the Transformer with early stopping
    4. Save the best model, scaler, and label encoder
    5. Plot training curves
"""

import argparse
import os
import sys
import joblib
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PISSGConfig, ModelConfig, TrainConfig, DataConfig
from src.preprocessing import load_and_preprocess
from src.pissg import create_geometric_sequential_data
from src.train import train_model, set_seed
from src.evaluate import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the T3C (Temporal Transformer for Thermal Comfort) model."
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the ASHRAE II CSV file.")
    parser.add_argument("--output_dir", type=str, default="saved_models",
                        help="Directory to save model weights and artifacts.")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)

    # PISSG
    parser.add_argument("--seq_length", type=int, default=12,
                        help="Number of timesteps in synthetic sequence.")

    # Training
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--jitter_std", type=float, default=0.005)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build configs from CLI arguments
    pissg_cfg = PISSGConfig(seq_length=args.seq_length)

    model_cfg = ModelConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seq_length=args.seq_length,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        jitter_std=args.jitter_std,
        early_stop_patience=args.early_stop_patience,
        random_seed=args.seed,
    )

    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Step 1: Preprocess
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)

    (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_enc, y_val_enc, y_test_enc,
        label_encoder, scaler, target_encoder,
    ) = load_and_preprocess(args.data_path, train_cfg=train_cfg)

    # Update input_dim based on actual feature count + deltas
    num_base_features = X_train_scaled.shape[1]
    num_transient = len([c for c in pissg_cfg.transient_cols if c in X_train_scaled.columns])
    model_cfg.input_dim = num_base_features + num_transient

    print(f"  Input dim: {num_base_features} base + {num_transient} deltas = {model_cfg.input_dim}")

    # ------------------------------------------------------------------
    # Step 2: Generate temporal sequences via PISSG
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: PISSG — Generating temporal sequences")
    print("=" * 60)

    # Use different seeds for train/val/test to avoid identical noise patterns
    X_train_seq = create_geometric_sequential_data(X_train_scaled, pissg_cfg, seed=args.seed)
    X_val_seq = create_geometric_sequential_data(X_val_scaled, pissg_cfg, seed=args.seed + 1)
    X_test_seq = create_geometric_sequential_data(X_test_scaled, pissg_cfg, seed=args.seed + 2)

    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Val sequences:   {X_val_seq.shape}")
    print(f"  Test sequences:  {X_test_seq.shape}")

    # ------------------------------------------------------------------
    # Step 3: Train the model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Training T3C model")
    print("=" * 60)

    model, history = train_model(
        X_train_seq, y_train_enc,
        X_val_seq, y_val_enc,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    # ------------------------------------------------------------------
    # Step 4: Save artifacts
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Saving artifacts")
    print("=" * 60)

    # Model weights
    model_path = os.path.join(args.output_dir, "t3c_best.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Model weights: {model_path}")

    # Scaler (needed for inference on new data)
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler:         {scaler_path}")

    # Label encoder (needed to decode predictions)
    le_path = os.path.join(args.output_dir, "label_encoder.pkl")
    joblib.dump(label_encoder, le_path)
    print(f"  Label encoder:  {le_path}")

    # Target encoder (needed for categorical features)
    te_path = os.path.join(args.output_dir, "target_encoder.pkl")
    joblib.dump(target_encoder, te_path)
    print(f"  Target encoder: {te_path}")

    # Save test data for later evaluation
    import numpy as np
    test_data_path = os.path.join(args.output_dir, "test_data.npz")
    np.savez(
        test_data_path,
        X_test_seq=X_test_seq,
        y_test_enc=y_test_enc,
    )
    print(f"  Test data:      {test_data_path}")

    # ------------------------------------------------------------------
    # Step 5: Plot training curves
    # ------------------------------------------------------------------
    curves_path = os.path.join("figures", "training_curves.png")
    os.makedirs("figures", exist_ok=True)
    plot_training_history(history, save_path=curves_path)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
