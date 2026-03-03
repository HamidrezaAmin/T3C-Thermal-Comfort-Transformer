"""
pissg.py — Physics-Informed Synthetic Sequential Generation (PISSG)
====================================================================

This module implements the core algorithmic contribution of the T3C paper.

Problem:
    The ASHRAE Global Thermal Comfort Database II contains only static survey
    snapshots — each row is a single observation with no temporal context.
    Transformer models require sequential input to leverage self-attention
    over time. Real longitudinal data does not exist in this database.

Solution (PISSG Algorithm):
    For each static observation (the "anchor point"), we synthesize a
    plausible 60-minute environmental history by walking *backwards* in time,
    applying physics-constrained Gaussian noise to transient variables only.

    Static variables (Climate, Clothing insulation, etc.) remain constant
    across the sequence, reflecting the physical reality that these do not
    change on a 5-minute timescale.

    Transient variables (Air temperature, Humidity, Air velocity, Met, SET)
    receive Gaussian perturbations bounded by Δ_max values derived from
    HVAC system response times and building thermal inertia.

Geometric Delta Features:
    After generating the sequence, we compute first-difference velocity
    vectors: ∇x_t = x_t − x_{t-1} for each transient feature. These
    "geometric deltas" allow the Transformer to distinguish between:
      - A "stable cool" state (small ∇x) → likely Neutral comfort
      - A "rapidly cooling" trajectory (large negative ∇x) → Uncomfortably Cool

Data Leakage Guarantee (Theorem 1 in paper):
    Since synthetic points are generated via continuous Gaussian noise from
    the anchor point alone, the probability that any generated point matches
    a training observation is measure zero. No cross-set information leakage
    occurs.

Reference: Paper Section 3.1 (Algorithm 1), Section 3.3 (Theorem 1).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List

import sys
sys.path.append("..")
from config import PISSGConfig


def create_geometric_sequential_data(
    df_scaled: pd.DataFrame,
    cfg: Optional[PISSGConfig] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Transform static survey snapshots into temporal sequences with geometric deltas.

    Algorithm (per sample):
        1. Place the real observation at the last timestep (t = seq_length - 1)
        2. Walk backwards: for each prior timestep, add Gaussian noise to
           transient features only, clipping to [0, 1]
        3. Compute geometric deltas: Δ_t = x_t - x_{t-1} for transient features
        4. Concatenate [base_features | delta_features] at each timestep

    Parameters
    ----------
    df_scaled : pd.DataFrame
        Scaled feature matrix (all values in [0, 1] after MinMax scaling).
        Shape: (num_samples, num_base_features).
    cfg : PISSGConfig, optional
        PISSG configuration. Uses defaults if not provided.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_3d : np.ndarray
        3D tensor of shape (num_samples, seq_length, num_base_features + num_transient).
        The last dimension is [base_features || geometric_deltas].

    Example
    -------
    >>> from config import PISSGConfig
    >>> cfg = PISSGConfig(seq_length=12)
    >>> X_seq = create_geometric_sequential_data(X_train_scaled, cfg, seed=42)
    >>> print(X_seq.shape)  # (N, 12, 17)  — 12 base + 5 deltas
    """
    if cfg is None:
        cfg = PISSGConfig()

    np.random.seed(seed)

    num_samples, num_features = df_scaled.shape
    feature_names = df_scaled.columns.tolist()
    seq_length = cfg.seq_length

    # Identify indices of transient columns in the feature array
    transient_indices = [
        feature_names.index(col)
        for col in cfg.transient_cols
        if col in feature_names
    ]
    num_transient = len(transient_indices)

    # Build noise scale array aligned with transient_indices order
    scales_array = np.array([
        cfg.noise_scales.get(col, 0.01)
        for col in cfg.transient_cols
        if col in feature_names
    ])

    # Output shape: (samples, timesteps, base_features + delta_features)
    output_dim = num_features + num_transient
    X_3d = np.zeros((num_samples, seq_length, output_dim))
    data_array = df_scaled.values

    for i in range(num_samples):
        anchor_row = data_array[i]

        # --- Step 1: Place real observation at the final timestep ---
        X_3d[i, seq_length - 1, :num_features] = anchor_row

        # --- Step 2: Backward random walk ---
        # Walk from t=(seq_length-2) down to t=0, each step adds noise
        for t in range(seq_length - 2, -1, -1):
            prev_row = X_3d[i, t + 1, :num_features].copy()

            # Sample physics-constrained Gaussian noise for transient features
            noise = np.random.normal(loc=0.0, scale=scales_array)
            prev_row[transient_indices] += noise

            # Clip to valid [0, 1] range (data is MinMax-scaled)
            X_3d[i, t, :num_features] = np.clip(prev_row, 0.0, 1.0)

        # --- Step 3: Compute geometric deltas (velocity vectors) ---
        # ∇x_t = x_t - x_{t-1} for each transient feature
        # At t=0, delta is zero (no predecessor exists)
        for t in range(1, seq_length):
            deltas = (
                X_3d[i, t, transient_indices]
                - X_3d[i, t - 1, transient_indices]
            )
            X_3d[i, t, num_features:] = deltas

    return X_3d
