"""Data preprocessing, validation, and pipeline preparation for CoolBuild AI."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.data_loader import (
        load_clean_dataset,
        get_feature_and_target_columns,
        derive_building_shape_groups,
        PRETTY_FEATURE_NAMES,
        PRETTY_TARGET_NAMES,
        PROJECT_ROOT
    )
except ImportError:
    from data_loader import (
        load_clean_dataset,
        get_feature_and_target_columns,
        derive_building_shape_groups,
        PRETTY_FEATURE_NAMES,
        PRETTY_TARGET_NAMES,
        PROJECT_ROOT
    )

FEATURE_CONSTRAINTS = {
    "relative_compactness": {"min": 0.62, "max": 0.98, "type": "continuous", "step": 0.01},
    "surface_area": {"min": 514.5, "max": 808.5, "type": "continuous", "step": 12.25},
    "wall_area": {"min": 245.0, "max": 416.5, "type": "continuous", "step": 12.25},
    "roof_area": {"min": 110.25, "max": 220.5, "type": "continuous", "step": 12.25},
    "overall_height": {"values": [3.5, 7.0], "type": "discrete"},
    "orientation": {"values": [2, 3, 4, 5], "labels": {2: "North", 3: "East", 4: "South", 5: "West"}, "type": "discrete"},
    "glazing_area": {"values": [0.0, 0.10, 0.25, 0.40], "type": "discrete"},
    "glazing_area_distribution": {
        "values": [0, 1, 2, 3, 4, 5],
        "labels": {0: "None (0%)", 1: "Uniform", 2: "North", 3: "East", 4: "South", 5: "West"},
        "type": "discrete"
    },
}


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform rigorous data quality verification:
    - Missing value detection
    - Duplicate detection
    - Data type verification
    - Range and boundary checks
    """
    features, targets = get_feature_and_target_columns()
    expected_cols = features + targets

    missing_counts = df.isnull().sum()
    missing_total = int(missing_counts.sum())
    
    duplicate_rows = int(df.duplicated().sum())
    feature_duplicates = int(df.duplicated(subset=features).sum())
    
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # Boundary violations
    range_violations = {}
    for feat in features:
        if feat in FEATURE_CONSTRAINTS:
            cfg = FEATURE_CONSTRAINTS[feat]
            if cfg["type"] == "continuous":
                viol = df[(df[feat] < cfg["min"]) | (df[feat] > cfg["max"])]
                if len(viol) > 0:
                    range_violations[feat] = len(viol)
            elif cfg["type"] == "discrete":
                viol = df[~df[feat].isin(cfg["values"])]
                if len(viol) > 0:
                    range_violations[feat] = len(viol)

    # Target sanity check (Loads must be strictly positive)
    target_violations = {}
    for tgt in targets:
        viol = df[df[tgt] <= 0]
        if len(viol) > 0:
            target_violations[tgt] = len(viol)

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values_total": missing_total,
        "missing_per_column": missing_counts.to_dict(),
        "duplicate_rows": duplicate_rows,
        "feature_duplicate_rows": feature_duplicates,
        "data_types": dtypes,
        "range_violations": range_violations,
        "target_violations": target_violations,
        "is_clean": (missing_total == 0 and duplicate_rows == 0 and len(range_violations) == 0 and len(target_violations) == 0)
    }
    return report


def get_feature_ranges(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Return actual min, max, mean, std for each feature in the dataset."""
    features, _ = get_feature_and_target_columns()
    ranges = {}
    for feat in features:
        ranges[feat] = {
            "min": float(df[feat].min()),
            "max": float(df[feat].max()),
            "mean": float(df[feat].mean()),
            "std": float(df[feat].std()),
            "unique_count": int(df[feat].nunique()),
            "values": sorted([float(x) for x in df[feat].unique()])
        }
    return ranges


def prepare_train_test_data(
    df: pd.DataFrame,
    target_name: str = "cooling_load",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets for a single target."""
    features, targets = get_feature_and_target_columns()
    if target_name not in targets:
        raise ValueError(f"Target '{target_name}' not in recognized targets {targets}")

    X = df[features].copy()
    y = df[target_name].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def get_scaled_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on training features and transform train and test sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    df = load_clean_dataset()
    report = validate_dataset(df)
    print("=== DATA QUALITY REPORT ===")
    print(f"Total Rows: {report['total_rows']}")
    print(f"Total Columns: {report['total_columns']}")
    print(f"Total Missing Values: {report['missing_values_total']}")
    print(f"Duplicate Full Rows: {report['duplicate_rows']}")
    print(f"Duplicate Feature Profiles: {report['feature_duplicate_rows']}")
    print(f"Range Violations: {report['range_violations']}")
    print(f"Target Violations: {report['target_violations']}")
    print(f"Dataset Clean & Valid: {report['is_clean']}")
