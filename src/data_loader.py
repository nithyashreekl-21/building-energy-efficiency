"""Data loading, acquisition, and validation module for UCI Energy Efficiency Dataset."""

import os
import zipfile
import urllib.request
import ssl
import pandas as pd
from typing import Tuple, Dict

# Canonical directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_ZIP_PATH = os.path.join(DATASET_DIR, "energy_efficiency.zip")
RAW_EXCEL_PATH = os.path.join(DATASET_DIR, "ENB2012_data.xlsx")
CSV_PATH = os.path.join(DATASET_DIR, "energy_efficiency.csv")
SOURCE_DOC_PATH = os.path.join(DATASET_DIR, "DATASET_SOURCE.md")

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
UCI_BACKUP_EXCEL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"

# Standard feature and target definitions
FEATURE_MAPPING = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
}

TARGET_MAPPING = {
    "Y1": "heating_load",
    "Y2": "cooling_load",
}

PRETTY_FEATURE_NAMES = {
    "relative_compactness": "Relative Compactness",
    "surface_area": "Surface Area (m²)",
    "wall_area": "Wall Area (m²)",
    "roof_area": "Roof Area (m²)",
    "overall_height": "Overall Height (m)",
    "orientation": "Orientation",
    "glazing_area": "Glazing Area (ratio)",
    "glazing_area_distribution": "Glazing Area Distribution",
}

PRETTY_TARGET_NAMES = {
    "heating_load": "Heating Load (kWh/m²)",
    "cooling_load": "Cooling Load (kWh/m²)",
}


def download_uci_dataset(force: bool = False) -> str:
    """Download the official UCI Energy Efficiency dataset if not already present."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    if not force and os.path.exists(RAW_EXCEL_PATH):
        print(f"Original dataset file found at {RAW_EXCEL_PATH}.")
        return RAW_EXCEL_PATH

    print(f"Downloading UCI Energy Efficiency dataset from {UCI_ZIP_URL}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        urllib.request.urlretrieve(UCI_ZIP_URL, RAW_ZIP_PATH)
        print("Extracting archive...")
        with zipfile.ZipFile(RAW_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print(f"Dataset extracted successfully to {DATASET_DIR}")
    except Exception as e:
        print(f"Zip download failed: {e}. Attempting direct Excel download...")
        urllib.request.urlretrieve(UCI_BACKUP_EXCEL_URL, RAW_EXCEL_PATH)

    # Locate extracted Excel file
    excel_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".xlsx")]
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx file found in {DATASET_DIR} after extraction.")
    
    final_excel_path = os.path.join(DATASET_DIR, excel_files[0])
    if final_excel_path != RAW_EXCEL_PATH:
        os.rename(final_excel_path, RAW_EXCEL_PATH)

    # Clean up zip archive if extracted
    if os.path.exists(RAW_ZIP_PATH):
        try:
            os.remove(RAW_ZIP_PATH)
        except OSError:
            pass

    # Generate source documentation
    create_dataset_documentation()
    return RAW_EXCEL_PATH


def create_dataset_documentation():
    """Create DATASET_SOURCE.md documenting provenance, citations, and structure."""
    doc_content = f"""# UCI Energy Efficiency Dataset Documentation

## 1. Provenance and Citation
- **Dataset Name**: Energy Efficiency Dataset
- **UCI Repository ID**: 242
- **Direct Source URL**: {UCI_ZIP_URL}
- **Original Authors**:
  - Angeliki Xifara (Civil Engineering, University of Cambridge)
  - Athanasios Tsanas (Oxford Centre for Industrial and Applied Mathematics, University of Oxford)
- **Publication**:
  - Tsanas, A., & Xifara, A. (2012). *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools*. Energy and Buildings, 49, 560-567.

## 2. Dataset Characteristics
- **Total Configurations**: 768 simulated residential building configurations
- **Origin**: Generated using Ecotect energy simulation software based on 12 distinct building shapes with 4 orientations, 4 glazing area ratios, and 5 glazing distributions.
- **Input Features**: 8 continuous/discrete parameters (X1 to X8)
- **Target Variables**: 2 continuous parameters (Y1 Heating Load, Y2 Cooling Load in kWh/m²)

## 3. Feature Descriptions
| Code | Feature Name | Units / Scale | Description |
| :--- | :--- | :--- | :--- |
| **X1** | Relative Compactness | Dimensionless (0.62 - 0.98) | Ratio of building volume to surface area relative to a reference cube |
| **X2** | Surface Area | m² (514.5 - 808.5) | Total external envelope surface area |
| **X3** | Wall Area | m² (245.0 - 416.5) | Total area of vertical external walls |
| **X4** | Roof Area | m² (110.25 - 220.5) | Horizontal roof envelope area |
| **X5** | Overall Height | m (3.5 or 7.0) | Total vertical height of the building (single vs two-story) |
| **X6** | Orientation | Categorical (2: North, 3: East, 4: South, 5: West) | Primary solar facing orientation |
| **X7** | Glazing Area | Ratio (0.0, 0.10, 0.25, 0.40) | Proportion of wall area allocated to windows/glazing |
| **X8** | Glazing Area Distribution | Categorical (0 to 5) | Distribution pattern: 0=None, 1=Uniform, 2=North, 3=East, 4=South, 5=West |

## 4. Target Variables
| Code | Target Name | Units | Description |
| :--- | :--- | :--- | :--- |
| **Y1** | Heating Load | kWh/m² (6.01 - 43.10) | Energy demand required to maintain indoor heating comfort |
| **Y2** | Cooling Load | kWh/m² (10.90 - 48.03) | Energy demand required to maintain indoor cooling comfort |

## 5. Geometric Structure & Grouping
The 768 samples are derived from **12 distinct building physical forms/shapes**. Each shape was simulated across 4 orientations and various glazing configurations (glazing area 0 has 1 sample per orientation; glazing areas 0.10, 0.25, 0.40 each have 5 glazing distributions per orientation).
- Geometry is uniquely identified by `(relative_compactness, surface_area, wall_area, roof_area, overall_height)`.
- Grouped Cross-Validation uses these 12 distinct shape clusters to prevent optimistic leakage between train and test splits.
"""
    with open(SOURCE_DOC_PATH, "w", encoding="utf-8") as f:
        f.write(doc_content)
    print(f"Documentation written to {SOURCE_DOC_PATH}")


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw dataset from Excel, ensuring download if needed."""
    if not os.path.exists(RAW_EXCEL_PATH):
        download_uci_dataset()

    df = pd.read_excel(RAW_EXCEL_PATH)
    # Some UCI versions include extra empty trailing rows or columns
    df = df.dropna(how="all")
    if len(df.columns) > 10:
        df = df.iloc[:, :10]
    return df


def load_clean_dataset(force_reload: bool = False) -> pd.DataFrame:
    """Load the standardized dataset with descriptive column names, caching to CSV."""
    if not force_reload and os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    df_raw = load_raw_dataset()
    # Normalize column names
    col_mapping = {}
    for col in df_raw.columns:
        col_str = str(col).strip()
        if col_str in FEATURE_MAPPING:
            col_mapping[col] = FEATURE_MAPPING[col_str]
        elif col_str in TARGET_MAPPING:
            col_mapping[col] = TARGET_MAPPING[col_str]
        else:
            # Map by index if headers don't match X1-X8, Y1-Y2 directly
            pass

    if len(col_mapping) == 10:
        df_clean = df_raw.rename(columns=col_mapping)
    else:
        # Fallback to positional mapping
        expected_cols = [
            "relative_compactness", "surface_area", "wall_area", "roof_area",
            "overall_height", "orientation", "glazing_area",
            "glazing_area_distribution", "heating_load", "cooling_load"
        ]
        df_clean = df_raw.iloc[:, :10].copy()
        df_clean.columns = expected_cols

    # Save standardized CSV
    df_clean.to_csv(CSV_PATH, index=False)
    print(f"Standardized clean dataset saved to {CSV_PATH} with {len(df_clean)} rows.")
    return df_clean


def get_feature_and_target_columns() -> Tuple[list, list]:
    """Return lists of feature column names and target column names."""
    feature_cols = [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
        "overall_height",
        "orientation",
        "glazing_area",
        "glazing_area_distribution",
    ]
    target_cols = ["heating_load", "cooling_load"]
    return feature_cols, target_cols


def derive_building_shape_groups(df: pd.DataFrame) -> pd.Series:
    """
    Derive the 12 original building shape groups strictly from geometric invariant features.
    
    The 12 building geometries are uniquely determined by:
    - relative_compactness
    - surface_area
    - wall_area
    - roof_area
    - overall_height
    
    Each of the 12 shapes has exactly 64 configurations:
    4 orientations * (1 [glazing=0] + 3 * 5 [glazing in {0.1, 0.25, 0.4} * 5 distributions]) = 4 * 16 = 64 rows.
    64 rows * 12 shapes = 768 total rows.
    """
    geom_cols = [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
        "overall_height",
    ]
    
    # Create deterministic shape IDs
    shape_df = df[geom_cols].drop_duplicates().sort_values(by=geom_cols).reset_index(drop=True)
    shape_df["shape_id"] = [f"Shape_{i+1:02d}" for i in range(len(shape_df))]
    
    merged = pd.merge(df, shape_df, on=geom_cols, how="left")
    return merged["shape_id"]


if __name__ == "__main__":
    print("Executing data_loader verification...")
    excel_path = download_uci_dataset()
    df = load_clean_dataset(force_reload=True)
    features, targets = get_feature_and_target_columns()
    print(f"Dataset Shape: {df.shape}")
    print(f"Features ({len(features)}): {features}")
    print(f"Targets ({len(targets)}): {targets}")
    
    shapes = derive_building_shape_groups(df)
    shape_counts = shapes.value_counts().sort_index()
    print("\nDerived Building Shape Distribution:")
    print(shape_counts)
    assert len(shape_counts) == 12, f"Expected 12 building shapes, found {len(shape_counts)}"
    assert all(count == 64 for count in shape_counts.values), "Each shape must contain exactly 64 configurations"
    print("\nVerification Passed: 12 building shapes identified with exactly 64 configurations each!")
