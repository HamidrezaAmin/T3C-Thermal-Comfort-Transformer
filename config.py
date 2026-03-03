"""
config.py — Centralized Hyperparameters for T3C
=================================================

All model, training, and PISSG parameters are defined here.
Override any value via command-line arguments in scripts/run_train.py.

Reference: Section 3.2 (Experimental Setup) of the paper.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PISSGConfig:
    """
    Physics-Informed Synthetic Sequential Generation (PISSG) parameters.

    The noise scales represent maximum physically plausible drift per 5-minute
    interval, grounded in HVAC system response times and building thermal inertia.
    See paper Section 3.1, Definition 2 for physical justifications.
    """

    seq_length: int = 12  # Number of timesteps (12 × 5 min = 60 min history)

    # Transient variables: these are allowed to drift in backward random walk.
    # Static variables (e.g., Climate, Building type) remain constant.
    transient_cols: List[str] = field(default_factory=lambda: [
        "SET",
        "Met",
        "Air temperature (°C)",
        "Relative humidity (%)",
        "Air velocity (m/s)",
    ])

    # Feature-specific noise scales (σ per 5-minute step).
    # Physical justification:
    #   - Air temperature: ~0.075°C drift reflects HVAC response time
    #   - Relative humidity: ~1.4% fluctuation per step
    #   - Air velocity: small jitter from turbulent airflow
    #   - Met (metabolic rate): minor activity-level changes
    #   - SET (Standard Effective Temperature): follows temperature drift
    noise_scales: Dict[str, float] = field(default_factory=lambda: {
        "Air temperature (°C)": 0.075,
        "Relative humidity (%)": 0.014,
        "Air velocity (m/s)": 0.02,
        "Met": 0.05,
        "SET": 0.005,
    })


@dataclass
class ModelConfig:
    """
    T3C (Temporal Transformer for Thermal Comfort) architecture parameters.

    Architecture: Linear Projection → Learnable Positional Embedding →
                  6× Transformer Encoder → Last-Token Pooling → Classifier

    Reference: Paper Section 3, Figure 1.
    """

    input_dim: int = 17       # 12 base features + 5 geometric delta features
    num_classes: int = 3      # Neutral (N), Uncomfortably Warm (UW), Uncomfortably Cool (UC)
    d_model: int = 256        # Transformer embedding dimension
    nhead: int = 8            # Number of self-attention heads
    num_layers: int = 6       # Number of Transformer encoder layers
    dim_feedforward: int = 512  # FFN hidden dimension inside each encoder layer
    dropout: float = 0.10     # Dropout rate (embedding, attention, FFN)
    seq_length: int = 12      # Must match PISSGConfig.seq_length


@dataclass
class TrainConfig:
    """Training loop parameters."""

    epochs: int = 55
    batch_size: int = 64
    learning_rate: float = 1e-4
    jitter_std: float = 0.005       # Gaussian noise σ for training-time augmentation
    weight_decay: float = 0.0       # Adam weight decay (L2 regularization)
    max_grad_norm: float = 1.0      # Gradient clipping (stabilizes Transformer training)
    scheduler_patience: int = 3     # ReduceLROnPlateau patience
    scheduler_factor: float = 0.5   # LR reduction factor
    early_stop_patience: int = 10   # Stop if val loss doesn't improve for N epochs
    test_size: float = 0.20         # Fraction of data held out for testing
    val_size: float = 0.15          # Fraction of training data used for validation
    random_seed: int = 42


@dataclass
class DataConfig:
    """Dataset and preprocessing parameters."""

    # Columns dropped during preprocessing (irrelevant to thermal comfort prediction)
    drop_columns: List[str] = field(default_factory=lambda: [
        "Year", "Age", "Country",
        "Cooling startegy_building level",
        "Sex", "Air movement preference",
        "Building type", "City",
    ])

    # Target column name in the ASHRAE II CSV
    target_col: str = "Thermal category"

    # Season mapping for cyclical encoding (Section 3, preprocessing)
    season_map: Dict[str, int] = field(default_factory=lambda: {
        "Winter": 1,
        "Spring": 2,
        "Summer": 3,
        "Autumn": 4,
    })
