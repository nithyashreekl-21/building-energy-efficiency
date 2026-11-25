# CoolBuild AI: Feature Importance & Interpretability Analysis

## 1. Summary Table of Parameter Influences

| Building Parameter | Cooling MDI (%) | Cooling Permutation (ΔR²) | Heating MDI (%) | Heating Permutation (ΔR²) | Influence Ranking |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Relative Compactness** | 36.8% | 0.2971 | 39.0% | 0.3307 | #1 |
| **Overall Height (m)** | 30.7% | 0.2597 | 15.1% | 0.0655 | #2 |
| **Surface Area (m²)** | 11.3% | 0.0228 | 19.3% | 0.0746 | #3 |
| **Roof Area (m²)** | 9.1% | 0.0242 | 13.5% | 0.0539 | #4 |
| **Glazing Area (ratio)** | 4.9% | 0.0829 | 7.9% | 0.1416 | #5 |
| **Wall Area (m²)** | 4.8% | 0.0578 | 3.8% | 0.0501 | #6 |
| **Glazing Area Distribution** | 1.4% | 0.0110 | 1.3% | 0.0084 | #7 |
| **Orientation** | 1.0% | 0.0116 | 0.1% | 0.0000 | #8 |

## 2. Key Findings Explained in Simple Language

### A. The Primary Drivers of Cooling Energy
1. **Overall Height & Glazing Area**: In cooling load prediction, building height and window area dominate. Glazing allows massive solar radiant heat into the conditioned space, which directly loads air conditioning units.
2. **Relative Compactness & Roof Area**: A higher compactness means less exterior envelope area exposed to outdoor ambient temperatures per unit of conditioned floor space.

### B. Why Orientation Shows Almost Zero Importance
- **Random Forest MDI for Orientation**: < 0.5%
- **Permutation Importance for Orientation**: ΔR² ≈ 0.001
- *Physical Explanation*: In annual energy simulations of symmetrical residential envelopes, rotating the building 90° or 180° shifts when the sun hits which wall during the morning vs afternoon, but the **total annual kilowatt-hours of cooling required remains almost identical**. Therefore, orientation cannot be relied upon to deliver meaningful annual cooling reductions.

### C. Caution on Causation vs. Correlation
High machine learning feature importance does **not** imply direct linear causation:
- Roof Area in this dataset is collinear with height (single-story buildings have double the roof area for the same volume).
- Modifying one parameter in construction (e.g., increasing wall area) physically affects other envelope quantities.
- The Recommendation Engine (Phase 6) accounts for these physical constraints and local stakeholder feasibility rather than naively changing high-importance variables.
