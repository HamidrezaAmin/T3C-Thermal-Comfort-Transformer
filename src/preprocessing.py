"""
preprocessing.py — Data Loading, Encoding, and Scaling Pipeline
================================================================

This module handles the full preprocessing pipeline for the ASHRAE Global
Thermal Comfort Database II:

    1. Load raw CSV and drop irrelevant columns
    2. Cyclical encoding for the Season feature (sin/cos projection)
    3. Target encoding for remaining categorical features
    4. Label encoding for the 3-class target variable
    5. Train/validation/test split with stratification
    6. MinMax scaling to [0, 1] range

Design Decisions:
    - Season is encoded cyclically because Winter→Spring→Summer→Autumn→Winter
      forms a circular continuum; ordinal encoding would incorrectly imply that
      Winter (1) and Autumn (4) are maximally distant.
    - Target encoding is used for high-cardinality categoricals (Climate, etc.)
      instead of one-hot encoding, which would create sparse, high-dimensional
      features that degrade Transformer attention efficiency.
    - MinMax scaling (not StandardScaler) is used because the PISSG algorithm
      clips generated values to [0, 1], and the Geometric Delta features are
      interpretable as normalized rates of change.

Reference: Paper Section 3 (Methodology).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from typing import Tuple, Dict, Optional

import sys
sys.path.append("..")
from config import DataConfig, TrainConfig


def encode_cyclical_season(df: pd.DataFrame, season_map: Dict[str, int]) -> pd.DataFrame:
    """
    Encode the 'Season' column as cyclical sin/cos features.

    Seasons form a circular continuum (Winter→Spring→Summer→Autumn→Winter),
    so we project them onto a unit circle using sine and cosine transforms.
    This preserves the fact that Winter and Autumn are adjacent, which ordinal
    encoding (1, 2, 3, 4) would not capture.

    Mapping:
        Winter=1 → (sin=1.0,  cos=0.0)   — top of circle
        Spring=2 → (sin=0.0,  cos=−1.0)  — left
        Summer=3 → (sin=−1.0, cos=0.0)   — bottom
        Autumn=4 → (sin=0.0,  cos=1.0)   — right

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'Season' column (string or numeric).
    season_map : dict
        Mapping from season names to integers {1, 2, 3, 4}.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Season' replaced by 'Season_sin' and 'Season_cos'.
    """
    df = df.copy()

    # Convert string labels to integers if necessary
    if df["Season"].dtype == "object":
        df["Season"] = df["Season"].map(season_map)

    # Project onto unit circle (period = 4 seasons)
    df["Season_sin"] = np.sin(2 * np.pi * df["Season"] / 4)
    df["Season_cos"] = np.cos(2 * np.pi * df["Season"] / 4)

    return df.drop(columns=["Season"])


def load_and_preprocess(
    data_path: str,
    data_cfg: Optional[DataConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    np.ndarray, np.ndarray, np.ndarray,
    LabelEncoder, MinMaxScaler, TargetEncoder
]:
    """
    Full preprocessing pipeline: load → clean → encode → split → scale.

    Parameters
    ----------
    data_path : str
        Path to the ASHRAE II CSV file.
    data_cfg : DataConfig, optional
        Dataset configuration. Uses defaults if not provided.
    train_cfg : TrainConfig, optional
        Training configuration (for split ratios and seed).

    Returns
    -------
    X_train_scaled : pd.DataFrame
        Scaled training features (for PISSG input).
    X_val_scaled : pd.DataFrame
        Scaled validation features.
    X_test_scaled : pd.DataFrame
        Scaled test features.
    y_train_enc : np.ndarray
        Label-encoded training targets.
    y_val_enc : np.ndarray
        Label-encoded validation targets.
    y_test_enc : np.ndarray
        Label-encoded test targets.
    label_encoder : LabelEncoder
        Fitted label encoder (for inverse_transform at evaluation time).
    scaler : MinMaxScaler
        Fitted scaler (needed to preprocess new data at inference time).
    target_encoder : TargetEncoder
        Fitted target encoder for categorical features.
    """
    if data_cfg is None:
        data_cfg = DataConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()

    # ------------------------------------------------------------------
    # 1. Load and clean
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path, encoding="latin1")

    # Drop columns that are not relevant to thermal comfort prediction.
    # These include demographic identifiers and building metadata that
    # would cause the model to memorize participant-specific patterns
    # rather than learning generalizable thermal dynamics.
    cols_to_drop = [c for c in data_cfg.drop_columns if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # ------------------------------------------------------------------
    # 2. Separate features and target
    # ------------------------------------------------------------------
    X = df.drop(columns=[data_cfg.target_col])
    y = df[data_cfg.target_col]

    # ------------------------------------------------------------------
    # 3. Train / Test split (stratified to preserve class distribution)
    # ------------------------------------------------------------------
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_seed,
        stratify=y,
    )

    # Further split training into train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=train_cfg.val_size,
        random_state=train_cfg.random_seed,
        stratify=y_train_full,
    )

    # ------------------------------------------------------------------
    # 4. Label-encode the target (N=0, UC=1, UW=2 or similar)
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    print(f"Class mapping: {dict(enumerate(le.classes_))}")

    # ------------------------------------------------------------------
    # 5. Cyclical season encoding
    # ------------------------------------------------------------------
    if "Season" in X_train.columns:
        X_train = encode_cyclical_season(X_train, data_cfg.season_map)
        X_val = encode_cyclical_season(X_val, data_cfg.season_map)
        X_test = encode_cyclical_season(X_test, data_cfg.season_map)

    # ------------------------------------------------------------------
    # 6. Target-encode remaining categorical columns
    # ------------------------------------------------------------------
    cat_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    te = TargetEncoder(cols=cat_cols)
    X_train = te.fit_transform(X_train, y_train_enc)
    X_val = te.transform(X_val)
    X_test = te.transform(X_test)

    # ------------------------------------------------------------------
    # 7. MinMax scaling to [0, 1]
    # ------------------------------------------------------------------
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
    )

    print(f"Preprocessing complete.")
    print(f"  Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"  Features: {X_train_scaled.columns.tolist()}")

    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_enc, y_val_enc, y_test_enc,
        le, scaler, te,
    )
