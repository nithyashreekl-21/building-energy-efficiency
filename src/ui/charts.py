"""Interactive Altair & Matplotlib chart generators for CoolBuild AI.
Implements the Climate-Tech aesthetic: white background, subtle gridlines, clean tooltips, proper units.
"""

import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, Any, List, Optional

# Climate-Tech SaaS Color Palette
PALETTE = {
    "primary": "#174D3A",       # Deep Forest Green
    "secondary": "#A8C3A0",     # Sage Green
    "accent": "#D5E8B5",        # Soft Lime
    "accent_dark": "#2E7D32",   # Medium Green
    "heating": "#B85D35",       # Warm Terracotta
    "neutral_light": "#F7F8F5", # Off-White
    "border": "#E2E8DE",        # Subtle Border
    "text_dark": "#202820",     # Dark Charcoal
    "text_muted": "#5E6D5E"     # Muted Green-Gray
}


def create_before_after_chart(
    orig_cooling: float,
    rec_cooling: float,
    orig_heating: float,
    rec_heating: float
) -> alt.Chart:
    """Create an interactive before-and-after load comparison bar chart."""
    data = pd.DataFrame([
        {"Category": "Cooling Load (Target)", "Design": "Original Design", "Load (kWh/m²)": orig_cooling},
        {"Category": "Cooling Load (Target)", "Design": "Recommended Alternative", "Load (kWh/m²)": rec_cooling},
        {"Category": "Heating Load (Coupled)", "Design": "Original Design", "Load (kWh/m²)": orig_heating},
        {"Category": "Heating Load (Coupled)", "Design": "Recommended Alternative", "Load (kWh/m²)": rec_heating},
    ])

    chart = alt.Chart(data).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=38).encode(
        x=alt.X("Design:N", title=None, axis=alt.Axis(labelAngle=0, labelFont="Plus Jakarta Sans", labelFontSize=11)),
        y=alt.Y("Load (kWh/m²):Q", title="Predicted Energy Load (kWh/m²)", axis=alt.Axis(gridColor="#EEF2EB", titleFont="Plus Jakarta Sans", titleFontSize=12)),
        color=alt.Color("Design:N", scale=alt.Scale(
            domain=["Original Design", "Recommended Alternative"],
            range=["#94A394", PALETTE["primary"]]
        ), legend=alt.Legend(title="Building Design", orient="top", titleFont="Plus Jakarta Sans")),
        column=alt.Column("Category:N", title=None, header=alt.Header(labelFont="Plus Jakarta Sans", labelFontSize=13, labelFontWeight="bold")),
        tooltip=[
            alt.Tooltip("Category:N", title="Load Type"),
            alt.Tooltip("Design:N", title="Building Version"),
            alt.Tooltip("Load (kWh/m²):Q", format=".2f", title="Energy Demand")
        ]
    ).properties(
        width=250,
        height=320,
        background="transparent"
    ).configure_view(
        strokeOpacity=0
    )

    return chart


def create_feature_importance_chart(df_imp: pd.DataFrame, target: str = "cooling_load") -> alt.Chart:
    """Create an interactive horizontal bar chart for feature importances with tooltips."""
    col_name = "cooling_mdi" if target == "cooling_load" else "heating_mdi"
    perm_col = "cooling_perm" if target == "cooling_load" else "heating_perm"
    bar_color = PALETTE["primary"] if target == "cooling_load" else PALETTE["heating"]
    label = "Cooling Load" if target == "cooling_load" else "Heating Load"

    df_plot = df_imp.copy()
    df_plot["Importance_Pct"] = df_plot[col_name] * 100
    df_plot["Permutation_Drop"] = df_plot[perm_col]

    chart = alt.Chart(df_plot).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, height=24).encode(
        x=alt.X("Importance_Pct:Q", title="Mean Decrease in Impurity (Gini Importance %)", axis=alt.Axis(gridColor="#EEF2EB", titleFontSize=11)),
        y=alt.Y("pretty_name:N", sort="-x", title="Building Parameter", axis=alt.Axis(labelFontSize=11)),
        color=alt.value(bar_color),
        tooltip=[
            alt.Tooltip("pretty_name:N", title="Parameter"),
            alt.Tooltip("Importance_Pct:Q", format=".2f", title="MDI Importance (%)"),
            alt.Tooltip("Permutation_Drop:Q", format=".4f", title="Permutation Drop (ΔR²)")
        ]
    ).properties(
        title=alt.TitleParams(
            text=f"Relative Parameter Influence on {label}",
            subtitle="Higher percentage indicates stronger impact on model energy predictions",
            font="Plus Jakarta Sans",
            fontSize=14,
            fontWeight="bold",
            color=PALETTE["primary"],
            subtitleColor="#5E6D5E"
        ),
        height=280,
        background="transparent"
    ).configure_view(
        strokeOpacity=0
    )

    return chart


def create_model_comparison_bar_chart(df_cv: pd.DataFrame, metric: str = "R2") -> alt.Chart:
    """Create an interactive comparison of Random CV vs Grouped CV across algorithms."""
    records = []
    
    for _, row in df_cv.iterrows():
        algo = row["Algorithm"].replace(" Regressor", "").replace(" (SVR)", "")
        if algo == "Mean Predictor Baseline":
            continue
            
        r_val = float(str(row["Random_CV_R2"]).split("±")[0].strip()) if metric == "R2" else float(str(row["Random_CV_MAE"]).split("±")[0].strip())
        g_val = float(str(row["Grouped_CV_R2"]).split("±")[0].strip()) if metric == "R2" else float(str(row["Grouped_CV_MAE"]).split("±")[0].strip())
        
        records.append({"Algorithm": algo, "Protocol": "Random 5-Fold CV (Optimistic)", "Value": r_val})
        records.append({"Algorithm": algo, "Protocol": "Grouped 5-Fold CV (Unseen Shapes)", "Value": g_val})
        
    df_plot = pd.DataFrame(records)
    
    y_title = "R² Determination Score" if metric == "R2" else "Mean Absolute Error (kWh/m²)"
    title_text = "Cross-Validation Generalization Comparison (R² Score)" if metric == "R2" else "Cross-Validation Error Comparison (MAE in kWh/m²)"

    chart = alt.Chart(df_plot).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Protocol:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("Value:Q", title=y_title, axis=alt.Axis(gridColor="#EEF2EB")),
        color=alt.Color("Protocol:N", scale=alt.Scale(
            domain=["Random 5-Fold CV (Optimistic)", "Grouped 5-Fold CV (Unseen Shapes)"],
            range=[PALETTE["primary"], PALETTE["secondary"]]
        ), legend=alt.Legend(title="Cross-Validation Protocol", orient="top")),
        column=alt.Column("Algorithm:N", title=None, header=alt.Header(labelFont="Plus Jakarta Sans", labelFontSize=12, labelFontWeight="bold")),
        tooltip=[
            alt.Tooltip("Algorithm:N", title="Algorithm"),
            alt.Tooltip("Protocol:N", title="Validation Type"),
            alt.Tooltip("Value:Q", format=".4f", title="Score")
        ]
    ).properties(
        title=alt.TitleParams(
            text=title_text,
            font="Plus Jakarta Sans",
            fontSize=14,
            fontWeight="bold",
            color=PALETTE["primary"]
        ),
        width=130,
        height=260,
        background="transparent"
    ).configure_view(
        strokeOpacity=0
    )

    return chart


def create_thermal_gauge_chart(cooling_load: float, max_range: float = 48.0) -> alt.Chart:
    """Create a sleek visual thermal load bar relative to dataset range."""
    pct = min(100.0, max(0.0, (cooling_load / max_range) * 100))
    df = pd.DataFrame([
        {"Type": "Current Load", "Value": cooling_load, "Max": max_range, "Pct": pct}
    ])

    base = alt.Chart(df).mark_bar(cornerRadius=8, height=18, color="#E2E8DE").encode(
        x=alt.value(320)
    )

    bar = alt.Chart(df).mark_bar(cornerRadius=8, height=18, color=PALETTE["primary"]).encode(
        x=alt.X("Value:Q", scale=alt.Scale(domain=[0, max_range]), title=None, axis=None)
    )

    return (base + bar).properties(width=320, height=25, background="transparent")
