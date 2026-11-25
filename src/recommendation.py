"""Building design recommendation and constrained optimization engine for CoolBuild AI."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple

try:
    from src.data_loader import (
        load_clean_dataset,
        get_feature_and_target_columns,
        PRETTY_FEATURE_NAMES,
        PRETTY_TARGET_NAMES,
        PROJECT_ROOT
    )
    from src.preprocessing import FEATURE_CONSTRAINTS
    from src.train import predict_loads
    from src.community_validation import get_active_changeable_parameters
except ImportError:
    from data_loader import (
        load_clean_dataset,
        get_feature_and_target_columns,
        PRETTY_FEATURE_NAMES,
        PRETTY_TARGET_NAMES,
        PROJECT_ROOT
    )
    from preprocessing import FEATURE_CONSTRAINTS
    from train import predict_loads
    from community_validation import get_active_changeable_parameters


def get_dataset_geometry_archetypes() -> pd.DataFrame:
    """
    Return the 12 physically coherent building geometry archetypes from the dataset.
    Every archetype satisfies the physical envelope conservation law:
    surface_area == 2 * roof_area + wall_area
    """
    df = load_clean_dataset()
    geom_cols = ["relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height"]
    archetypes = df[geom_cols].drop_duplicates().sort_values(by=geom_cols).reset_index(drop=True)
    return archetypes


def generate_feasible_alternatives(
    initial_config: Dict[str, float],
    changeable_params: List[str]
) -> List[Dict[str, float]]:
    """
    Generate physically valid and realistic alternative building configurations
    strictly modifying ONLY parameters approved in changeable_params.
    """
    features, _ = get_feature_and_target_columns()
    changeable_set = set(changeable_params)
    
    geom_features = {"relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height"}
    has_geom_changes = bool(geom_features.intersection(changeable_set))
    
    # 1. Determine base geometry candidates
    archetypes = get_dataset_geometry_archetypes()
    
    if has_geom_changes:
        # Filter archetypes by any geometry parameter that is FIXED
        filtered_archetypes = archetypes.copy()
        for gf in geom_features:
            if gf not in changeable_set:
                filtered_archetypes = filtered_archetypes[filtered_archetypes[gf] == initial_config[gf]]
        if len(filtered_archetypes) == 0:
            # Fallback to current geometry if constraints over-constrain
            geometry_candidates = [
                {gf: initial_config[gf] for gf in geom_features}
            ]
        else:
            geometry_candidates = filtered_archetypes.to_dict(orient="records")
    else:
        # Geometry is completely fixed
        geometry_candidates = [
            {gf: initial_config[gf] for gf in geom_features}
        ]
        
    # 2. Determine Orientation candidates
    if "orientation" in changeable_set:
        orientation_candidates = [2, 3, 4, 5]
    else:
        orientation_candidates = [int(initial_config["orientation"])]
        
    # 3. Determine Glazing Area candidates
    if "glazing_area" in changeable_set:
        glazing_area_candidates = [0.0, 0.10, 0.25, 0.40]
    else:
        glazing_area_candidates = [float(initial_config["glazing_area"])]
        
    # 4. Determine Glazing Distribution candidates
    if "glazing_area_distribution" in changeable_set:
        glazing_dist_candidates = [0, 1, 2, 3, 4, 5]
    else:
        glazing_dist_candidates = [int(initial_config["glazing_area_distribution"])]
        
    # 5. Cross-generate candidates and enforce constraints
    candidates = []
    seen = set()
    
    for geom in geometry_candidates:
        for ori in orientation_candidates:
            for g_area in glazing_area_candidates:
                for g_dist in glazing_dist_candidates:
                    # Enforce constraint: if glazing_area is 0, distribution must be 0
                    actual_dist = 0 if g_area == 0.0 else (g_dist if g_dist != 0 else 1)
                    
                    candidate = {
                        "relative_compactness": float(geom["relative_compactness"]),
                        "surface_area": float(geom["surface_area"]),
                        "wall_area": float(geom["wall_area"]),
                        "roof_area": float(geom["roof_area"]),
                        "overall_height": float(geom["overall_height"]),
                        "orientation": int(ori),
                        "glazing_area": float(g_area),
                        "glazing_area_distribution": int(actual_dist)
                    }
                    
                    # Deduplicate
                    key = tuple(candidate[f] for f in features)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(candidate)
                        
    return candidates


def optimize_building_design(
    initial_config: Dict[str, float],
    changeable_params: Optional[List[str]] = None,
    algorithm: Optional[str] = "random_forest"
) -> Dict[str, Any]:
    """
    Main Recommendation Engine:
    1. Evaluates baseline building configuration.
    2. Identifies approved changeable parameters.
    3. Generates feasible alternative candidate configurations.
    4. Predicts energy loads for each candidate using the production ML model.
    5. Identifies the optimal configuration minimizing cooling load.
    6. Calculates savings and generates an engineering explanation.
    """
    features, _ = get_feature_and_target_columns()
    
    # Clean initial input
    clean_initial = {f: float(initial_config[f]) for f in features}
    
    # Determine changeable parameters
    active_changeable, active_fixed, is_provisional, latest_stakeholder = get_active_changeable_parameters()
    if changeable_params is not None:
        user_changeable = [p for p in changeable_params if p in features]
        if user_changeable:
            active_changeable = user_changeable
            active_fixed = [p for p in features if p not in active_changeable]
            
    # Baseline predictions
    original_loads = predict_loads(clean_initial, algorithm=algorithm)
    orig_cooling = original_loads["cooling_load"]
    orig_heating = original_loads["heating_load"]
    
    # Generate alternatives
    candidates = generate_feasible_alternatives(clean_initial, active_changeable)
    
    # Evaluate candidates
    scored_candidates = []
    for cand in candidates:
        loads = predict_loads(cand, algorithm=algorithm)
        cooling = loads["cooling_load"]
        heating = loads["heating_load"]
        reduction = orig_cooling - cooling
        reduction_pct = (reduction / orig_cooling) * 100 if orig_cooling > 0 else 0.0
        
        scored_candidates.append({
            "config": cand,
            "cooling_load": cooling,
            "heating_load": heating,
            "reduction": round(reduction, 2),
            "reduction_pct": round(reduction_pct, 2)
        })
        
    # Sort by cooling load ascending (lowest cooling load first)
    scored_candidates.sort(key=lambda x: x["cooling_load"])
    
    best_candidate = scored_candidates[0]
    best_config = best_candidate["config"]
    best_cooling = best_candidate["cooling_load"]
    best_heating = best_candidate["heating_load"]
    
    cooling_reduction = round(orig_cooling - best_cooling, 2)
    cooling_reduction_pct = round((cooling_reduction / orig_cooling) * 100, 2) if orig_cooling > 0 else 0.0
    heating_difference = round(best_heating - orig_heating, 2)
    
    # Identify specifically modified parameters
    modified_parameters = {}
    for f in features:
        if clean_initial[f] != best_config[f]:
            modified_parameters[f] = {
                "parameter_name": PRETTY_FEATURE_NAMES.get(f, f),
                "original_value": clean_initial[f],
                "recommended_value": best_config[f],
                "change": round(best_config[f] - clean_initial[f], 3)
            }
            
    # Formulate engineering explanation
    explanation = generate_engineering_explanation(
        clean_initial, best_config, modified_parameters, cooling_reduction, cooling_reduction_pct
    )
    
    return {
        "original_config": clean_initial,
        "recommended_config": best_config,
        "original_loads": original_loads,
        "recommended_loads": {
            "heating_load": best_heating,
            "cooling_load": best_cooling
        },
        "cooling_reduction": max(0.0, cooling_reduction),
        "cooling_reduction_pct": max(0.0, cooling_reduction_pct),
        "heating_difference": heating_difference,
        "modified_parameters": modified_parameters,
        "fixed_parameters": active_fixed,
        "changeable_parameters": active_changeable,
        "is_provisional": is_provisional,
        "latest_stakeholder": latest_stakeholder,
        "engineering_explanation": explanation,
        "total_candidates_evaluated": len(scored_candidates),
        "top_alternatives": scored_candidates[:5]
    }


def generate_engineering_explanation(
    orig: Dict[str, float],
    rec: Dict[str, float],
    mods: Dict[str, Any],
    reduction: float,
    reduction_pct: float
) -> str:
    """Generate practical, jargon-free civil & architectural explanation."""
    if reduction <= 0.05 or len(mods) == 0:
        return (
            "The baseline building configuration is already optimal within the allowed changeable parameters. "
            "No feasible adjustments produced a significant cooling reduction under current site constraints."
        )
        
    points = []
    
    if "glazing_area" in mods:
        orig_g = orig["glazing_area"]
        rec_g = rec["glazing_area"]
        if rec_g < orig_g:
            points.append(
                f"Reducing the window glazing area from {int(orig_g*100)}% to {int(rec_g*100)}% of wall area "
                f"significantly lowers the solar heat gain coefficient (SHGC) and direct radiative heat influx, "
                f"lessening the cooling demand."
            )
        elif rec_g > orig_g:
            points.append(f"Adjusting window proportion to {int(rec_g*100)}% optimized daytime natural ventilation pathways.")
            
    if "glazing_area_distribution" in mods:
        orig_dist = int(orig["glazing_area_distribution"])
        rec_dist = int(rec["glazing_area_distribution"])
        dist_labels = {0: "None", 1: "Uniform", 2: "North", 3: "East", 4: "South", 5: "West"}
        points.append(
            f"Relocating window distribution from '{dist_labels.get(orig_dist, str(orig_dist))}' to "
            f"'{dist_labels.get(rec_dist, str(rec_dist))}' minimizes peak afternoon solar exposure."
        )
        
    if "relative_compactness" in mods or "surface_area" in mods:
        orig_c = orig["relative_compactness"]
        rec_c = rec["relative_compactness"]
        if rec_c > orig_c:
            points.append(
                f"Increasing building relative compactness from {orig_c:.2f} to {rec_c:.2f} minimizes the exterior "
                f"surface area relative to indoor conditioned volume, reducing ambient heat conduction through walls and roof."
            )
            
    summary = (
        f"By implementing these approved modifications, the building's predicted cooling load decreases by "
        f"{reduction:.2f} kWh/m² (a {reduction_pct:.1f}% potential reduction). "
        + " ".join(points)
    )
    return summary


def generate_comparison_plot(
    opt_res: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate an attractive comparative before-and-after visualization
    contrasting original vs recommended building energy loads and changed features.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Bar comparison of Cooling and Heating loads
    categories = ["Cooling Load\n(Target to Reduce)", "Heating Load\n(Coupled Demand)"]
    orig_vals = [opt_res["original_loads"]["cooling_load"], opt_res["original_loads"]["heating_load"]]
    rec_vals = [opt_res["recommended_loads"]["cooling_load"], opt_res["recommended_loads"]["heating_load"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, orig_vals, width, label="Original Building", color="#95a5a6", edgecolor="black", alpha=0.85)
    ax1.bar(x + width/2, rec_vals, width, label="Recommended Design", color="#27ae60", edgecolor="black", alpha=0.85)
    
    ax1.set_ylabel("Predicted Energy Load (kWh/m²)", fontsize=11)
    ax1.set_title("Energy Load Comparison: Before vs. After", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    for p in ax1.patches:
        h = p.get_height()
        if h > 0:
            ax1.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points', fontweight="bold")
            
    # 2. Percentage reduction summary & modified parameter table
    ax2.axis("off")
    
    reduction_val = opt_res["cooling_reduction"]
    reduction_pct = opt_res["cooling_reduction_pct"]
    is_prov = opt_res.get("is_provisional", True)
    
    status_text = "⚠ PROVISIONAL (Requires Local Stakeholder Sign-Off)" if is_prov else "✔ VALIDATED BY LOCAL STAKEHOLDER"
    status_color = "#d35400" if is_prov else "#27ae60"
    
    ax2.text(0.05, 0.95, f"Predicted Cooling Reduction: {reduction_val:.2f} kWh/m²", fontsize=13, fontweight="bold", color="#2c3e50")
    ax2.text(0.05, 0.86, f"Percentage Potential Savings: {reduction_pct:.1f}%", fontsize=15, fontweight="bold", color="#27ae60")
    ax2.text(0.05, 0.77, f"Status: {status_text}", fontsize=10, fontweight="bold", color=status_color)
    
    # Table of changes
    mods = opt_res.get("modified_parameters", {})
    if mods:
        ax2.text(0.05, 0.65, "Modified Feasible Parameters:", fontsize=11, fontweight="bold")
        y_pos = 0.55
        for feat, data in mods.items():
            name = data["parameter_name"]
            orig_v = data["original_value"]
            rec_v = data["recommended_value"]
            chg = data["change"]
            ax2.text(0.08, y_pos, f"• {name}: {orig_v} → {rec_v}  (Δ {chg:+})", fontsize=10, color="#34495e")
            y_pos -= 0.09
    else:
        ax2.text(0.05, 0.60, "No parameter modifications required; baseline is already optimal.", fontsize=10, color="#7f8c8d")
        
    ax2.text(0.05, 0.05, "* All figures represent model-predicted thermal load reductions based on the UCI dataset.",
             fontsize=8, style="italic", color="#7f8c8d")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return fig


if __name__ == "__main__":
    print("Testing Building Design Recommendation Engine...")
    
    # Test initial baseline: 2-story building with 40% glazing
    sample_building = {
        "relative_compactness": 0.76,
        "surface_area": 661.5,
        "wall_area": 416.5,
        "roof_area": 122.5,
        "overall_height": 7.0,
        "orientation": 2,
        "glazing_area": 0.40,
        "glazing_area_distribution": 2
    }
    
    # Run optimization with only glazing changeable (standard practical scenario)
    res = optimize_building_design(sample_building, changeable_params=["glazing_area", "glazing_area_distribution"])
    
    print("\n=== OPTIMIZATION RESULT ===")
    print(f"Original Cooling Load:     {res['original_loads']['cooling_load']:.2f} kWh/m²")
    print(f"Recommended Cooling Load:  {res['recommended_loads']['cooling_load']:.2f} kWh/m²")
    print(f"Predicted Reduction:       {res['cooling_reduction']:.2f} kWh/m² ({res['cooling_reduction_pct']:.1f}%)")
    print(f"Provisional Status:        {res['is_provisional']}")
    print(f"\nModified Parameters ({len(res['modified_parameters'])}):")
    for k, v in res["modified_parameters"].items():
        print(f"  - {v['parameter_name']}: {v['original_value']} -> {v['recommended_value']} (Δ {v['change']:+})")
    print(f"\nEngineering Explanation:\n{res['engineering_explanation']}")
    
    # Test plot generation
    plot_out = os.path.join(PROJECT_ROOT, "results", "plots", "16_sample_building_optimization_comparison.png")
    generate_comparison_plot(res, save_path=plot_out)
    print(f"\nSaved test comparison plot to: {plot_out}")
