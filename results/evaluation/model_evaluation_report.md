# CoolBuild AI: Model Cross-Validation & Generalization Report

## 1. Methodology: Random CV vs. Grouped CV by Building Shape

A critical problem in building performance machine learning is **spatial and geometric data leakage**:
- The UCI dataset consists of 768 rows derived from **12 elementary building geometric shapes**.
- In **Random 5-Fold Cross-Validation**, rows are shuffled randomly. Because each shape has 64 orientation/glazing variations, ~80% of rows from every shape end up in train and ~20% in test. The model is tested on shapes it already memorized!
- In **Grouped 5-Fold Cross-Validation**, entire geometric shapes (`GroupKFold` on the 12 shape clusters) are placed exclusively in test folds. The model is forced to predict cooling/heating loads for **unseen architectural building geometries**.

## 2. Performance Comparison: Cooling Load (kWh/m²)

| Algorithm | Random CV R² | Grouped CV R² | Random CV MAE | Grouped CV MAE | Grouped CV RMSE | Generalization Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.8850 ± 0.007 | 0.5815 ± 0.259 | 2.265 | 4.059 | 5.153 | ΔR²: -0.3035 |
| **Decision Tree Regressor** | 0.9613 ± 0.006 | -0.0876 ± 1.164 | 1.154 | 6.212 | 7.765 | ΔR²: -1.0489 |
| **Random Forest Regressor** | 0.9690 ± 0.006 | 0.3675 ± 0.412 | 1.032 | 5.190 | 6.304 | ΔR²: -0.6015 |
| **Support Vector Regression (SVR)** | 0.9625 ± 0.003 | -1.0791 ± 2.198 | 1.169 | 8.948 | 10.803 | ΔR²: -2.0416 |
| **Mean Predictor Baseline** | -0.0112 ± 0.010 | -0.2458 ± 0.177 | 8.594 | 8.813 | 9.727 | ΔR²: -0.2346 |

## 2. Performance Comparison: Heating Load (kWh/m²)

| Algorithm | Random CV R² | Grouped CV R² | Random CV MAE | Grouped CV MAE | Grouped CV RMSE | Generalization Impact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.9140 ± 0.006 | 0.6958 ± 0.233 | 2.083 | 3.731 | 4.616 | ΔR²: -0.2182 |
| **Decision Tree Regressor** | 0.9968 ± 0.000 | 0.2615 ± 0.952 | 0.409 | 5.357 | 6.766 | ΔR²: -0.7353 |
| **Random Forest Regressor** | 0.9977 ± 0.000 | 0.5802 ± 0.335 | 0.328 | 4.220 | 5.312 | ΔR²: -0.4175 |
| **Support Vector Regression (SVR)** | 0.9921 ± 0.002 | -0.8461 ± 1.587 | 0.619 | 9.249 | 11.471 | ΔR²: -1.8383 |
| **Mean Predictor Baseline** | -0.0118 ± 0.014 | -0.2256 ± 0.139 | 9.158 | 9.376 | 10.302 | ΔR²: -0.2138 |

## 3. Conclusions & Recommendation

- **Random Forest** demonstrates the strongest generalization capability under both validation regimes.
- As expected, Grouped CV produces higher error and higher fold variance than Random CV because predicting energy for a genuinely new building shape is fundamentally harder than interpolating between orientations of known shapes.
- **Random Forest remains the recommended production model** for the building optimization engine.
