# UCI Energy Efficiency Dataset Documentation

## 1. Provenance and Citation
- **Dataset Name**: Energy Efficiency Dataset
- **UCI Repository ID**: 242
- **Direct Source URL**: https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip
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
