"""
T3C: Temporal Transformer for Thermal Comfort
==============================================

Modules:
    preprocessing  — Data loading, encoding, and scaling pipeline
    pissg          — Physics-Informed Synthetic Sequential Generation
    model          — T3C Transformer architecture
    train          — Training loop with early stopping
    evaluate       — Metrics, confusion matrix, and classification report
"""

from .model import ComfortTransformer
from .pissg import create_geometric_sequential_data
from .preprocessing import load_and_preprocess, encode_cyclical_season
from .train import train_model
from .evaluate import evaluate_model
