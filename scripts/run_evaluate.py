"""
run_evaluate.py — Standalone Evaluation Script
================================================

Usage:
    python scripts/run_evaluate.py --data_path data/df1.csv --model_path saved_models/t3c_best.pth
    python scripts/run_evaluate.py --test_data saved_models/test_data.npz --model_path saved_models/t3c_best.pth

Evaluates a trained T3C model on the held-out test set and produces:
    - Test accuracy and loss
    - Macro-averaged F1 score
    - Per-class precision, recall, and F1
    - Confusion matrix heatmap
"""

import argparse
import os
import sys
import joblib
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig, PISSGConfig, TrainConfig
from src.model import ComfortTransformer
from src.evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained T3C model on the test set."
    )

    # Option A: use pre-saved test data
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test_data.npz (saved during training).")

    # Option B: reprocess from raw CSV
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to raw ASHRAE II CSV (reprocesses from scratch).")

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model weights (.pth file).")
    parser.add_argument("--model_dir", type=str, default="saved_models",
                        help="Directory containing scaler.pkl, label_encoder.pkl.")

    # Architecture (must match training)
    parser.add_argument("--input_dim", type=int, default=17)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--seq_length", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    if args.test_data:
        # Fast path: load pre-saved test sequences
        print(f"Loading pre-saved test data from: {args.test_data}")
        data = np.load(args.test_data)
        X_test_seq = data["X_test_seq"]
        y_test_enc = data["y_test_enc"]
    elif args.data_path:
        # Full path: reprocess from raw CSV
        print(f"Reprocessing test data from: {args.data_path}")
        from src.preprocessing import load_and_preprocess
        from src.pissg import create_geometric_sequential_data

        _, _, X_test_scaled, _, _, y_test_enc, _, _, _ = load_and_preprocess(
            args.data_path
        )
        pissg_cfg = PISSGConfig(seq_length=args.seq_length)
        X_test_seq = create_geometric_sequential_data(X_test_scaled, pissg_cfg, seed=44)
    else:
        raise ValueError("Provide either --test_data or --data_path.")

    # ------------------------------------------------------------------
    # Load label encoder for class names
    # ------------------------------------------------------------------
    le_path = os.path.join(args.model_dir, "label_encoder.pkl")
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)
        class_names = list(label_encoder.classes_)
    else:
        print(f"Warning: label_encoder.pkl not found at {le_path}. Using default names.")
        class_names = ["Neutral", "Uncomfortably Cool", "Uncomfortably Warm"]

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_cfg = ModelConfig(
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
    )

    model = ComfortTransformer(model_cfg=model_cfg).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from: {args.model_path}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    metrics = evaluate_model(
        model=model,
        X_test_seq=X_test_seq,
        y_test_enc=y_test_enc,
        class_names=class_names,
        batch_size=args.batch_size,
        device=device,
        save_path="figures/confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
