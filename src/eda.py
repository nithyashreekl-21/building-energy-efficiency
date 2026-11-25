"""Exploratory Data Analysis (EDA) and visualization generator for CoolBuild AI."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")

# Set aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics including skewness and kurtosis."""
    desc = df.describe().T
    desc["skewness"] = df.skew()
    desc["kurtosis"] = df.kurtosis()
    desc["range"] = desc["max"] - desc["min"]
    desc["IQR"] = desc["75%"] - desc["25%"]
    
    os.makedirs(EVAL_DIR, exist_ok=True)
    out_csv = os.path.join(EVAL_DIR, "descriptive_statistics.csv")
    desc.to_csv(out_csv)
    print(f"Descriptive statistics saved to {out_csv}")
    return desc


def plot_feature_distributions(df: pd.DataFrame) -> str:
    """Generate 8-panel distribution plot for all input features."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    features, _ = get_feature_and_target_columns()
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    palette = sns.color_palette("mako", n_colors=8)
    
    for i, feat in enumerate(features):
        ax = axes[i]
        color = palette[i]
        pretty_name = PRETTY_FEATURE_NAMES.get(feat, feat)
        
        if df[feat].nunique() <= 6:
            # Discrete feature - bar plot
            counts = df[feat].value_counts().sort_index()
            sns.barplot(x=counts.index, y=counts.values, ax=ax, color=color, edgecolor="black", alpha=0.85)
            ax.set_ylabel("Frequency", fontsize=11)
            for p in ax.patches:
                ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
        else:
            # Continuous feature - histogram + KDE
            sns.histplot(df[feat], kde=True, ax=ax, color=color, edgecolor="black", alpha=0.6, bins=15)
            ax.set_ylabel("Count", fontsize=11)
            
        ax.set_title(f"{pretty_name}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(pretty_name, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("CoolBuild AI – Input Feature Distributions (N = 768)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "01_feature_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_target_distributions(df: pd.DataFrame) -> str:
    """Generate side-by-side distribution and boxplots for Heating and Cooling Loads."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Heating Load
    sns.histplot(df["heating_load"], kde=True, ax=axes[0, 0], color="#e74c3c", bins=25, edgecolor="black", alpha=0.65)
    axes[0, 0].set_title(f"Heating Load Distribution\nMean: {df['heating_load'].mean():.2f} | Median: {df['heating_load'].median():.2f} kWh/m²",
                         fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Heating Load (kWh/m²)", fontsize=11)
    axes[0, 0].set_ylabel("Count", fontsize=11)
    
    sns.boxplot(x=df["heating_load"], ax=axes[1, 0], color="#e74c3c", boxprops=dict(alpha=0.7))
    axes[1, 0].set_xlabel("Heating Load (kWh/m²)", fontsize=11)
    
    # Cooling Load
    sns.histplot(df["cooling_load"], kde=True, ax=axes[0, 1], color="#3498db", bins=25, edgecolor="black", alpha=0.65)
    axes[0, 1].set_title(f"Cooling Load Distribution\nMean: {df['cooling_load'].mean():.2f} | Median: {df['cooling_load'].median():.2f} kWh/m²",
                         fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Cooling Load (kWh/m²)", fontsize=11)
    axes[0, 1].set_ylabel("Count", fontsize=11)
    
    sns.boxplot(x=df["cooling_load"], ax=axes[1, 1], color="#3498db", boxprops=dict(alpha=0.7))
    axes[1, 1].set_xlabel("Cooling Load (kWh/m²)", fontsize=11)
    
    plt.suptitle("CoolBuild AI – Target Variable Energy Load Distributions", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "02_target_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Generate Pearson correlation heatmap showing inter-feature and feature-target relationships."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    features, targets = get_feature_and_target_columns()
    
    # Pretty column labels for readability
    rename_dict = {**PRETTY_FEATURE_NAMES, **PRETTY_TARGET_NAMES}
    df_renamed = df[features + targets].rename(columns=rename_dict)
    
    corr = df_renamed.corr()
    
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.75,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        annot_kws={"size": 9}
    )
    plt.title("CoolBuild AI – Feature & Target Correlation Matrix", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "03_correlation_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_feature_vs_cooling_load(df: pd.DataFrame) -> str:
    """Generate 8-panel plot showing each feature's direct relationship with Cooling Load."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    features, _ = get_feature_and_target_columns()
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        ax = axes[i]
        pretty_name = PRETTY_FEATURE_NAMES.get(feat, feat)
        
        if df[feat].nunique() <= 6:
            # Discrete feature - Boxplot
            sns.boxplot(x=feat, y="cooling_load", hue=feat, data=df, ax=ax, palette="Blues_d", legend=False, boxprops=dict(alpha=0.8))
            sns.stripplot(x=feat, y="cooling_load", data=df, ax=ax, color="black", alpha=0.25, jitter=0.2, size=3)
        else:
            # Continuous feature - Scatter with trendline
            sns.regplot(
                x=feat, y="cooling_load", data=df, ax=ax,
                scatter_kws={"alpha": 0.45, "color": "#1f77b4", "s": 25},
                line_kws={"color": "#d62728", "linewidth": 2}
            )
            
        r_val = df[feat].corr(df["cooling_load"])
        ax.set_title(f"{pretty_name} vs Cooling Load  (r = {r_val:+.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel(pretty_name, fontsize=10)
        ax.set_ylabel("Cooling Load (kWh/m²)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("CoolBuild AI – Building Design Parameters vs Cooling Load", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "04_feature_vs_cooling_load.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_feature_vs_heating_load(df: pd.DataFrame) -> str:
    """Generate 8-panel plot showing each feature's direct relationship with Heating Load."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    features, _ = get_feature_and_target_columns()
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        ax = axes[i]
        pretty_name = PRETTY_FEATURE_NAMES.get(feat, feat)
        
        if df[feat].nunique() <= 6:
            sns.boxplot(x=feat, y="heating_load", hue=feat, data=df, ax=ax, palette="Reds_d", legend=False, boxprops=dict(alpha=0.8))
            sns.stripplot(x=feat, y="heating_load", data=df, ax=ax, color="black", alpha=0.25, jitter=0.2, size=3)
        else:
            sns.regplot(
                x=feat, y="heating_load", data=df, ax=ax,
                scatter_kws={"alpha": 0.45, "color": "#e74c3c", "s": 25},
                line_kws={"color": "#2c3e50", "linewidth": 2}
            )
            
        r_val = df[feat].corr(df["heating_load"])
        ax.set_title(f"{pretty_name} vs Heating Load  (r = {r_val:+.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel(pretty_name, fontsize=10)
        ax.set_ylabel("Heating Load (kWh/m²)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("CoolBuild AI – Building Design Parameters vs Heating Load", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "05_feature_vs_heating_load.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_heating_vs_cooling_coupling(df: pd.DataFrame) -> str:
    """Plot Heating Load vs Cooling Load coupling, demonstrating close thermal affinity."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 7))
    
    scatter = sns.scatterplot(
        x="heating_load",
        y="cooling_load",
        hue="overall_height",
        palette={3.5: "#2ecc71", 7.0: "#9b59b6"},
        style="overall_height",
        data=df,
        s=60,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5
    )
    
    # Regression line
    sns.regplot(
        x="heating_load",
        y="cooling_load",
        data=df,
        scatter=False,
        ax=scatter,
        line_kws={"color": "#e67e22", "linestyle": "--", "linewidth": 2, "label": "Linear Fit"}
    )
    
    r_val = df["heating_load"].corr(df["cooling_load"])
    plt.title(f"Thermal Energy Coupling: Heating vs Cooling Load (r = {r_val:.3f})", fontsize=14, fontweight="bold")
    plt.xlabel("Heating Load (kWh/m²)", fontsize=12)
    plt.ylabel("Cooling Load (kWh/m²)", fontsize=12)
    plt.legend(title="Building Height (m)", frameon=True, facecolor="white", edgecolor="#ccc")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    out_path = os.path.join(PLOTS_DIR, "06_heating_vs_cooling_coupling.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_shape_groups_energy(df: pd.DataFrame) -> str:
    """Plot energy demand profiles across all 12 building shape groups."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df_copy = df.copy()
    df_copy["shape_id"] = derive_building_shape_groups(df)
    
    shape_agg = df_copy.groupby("shape_id")[["heating_load", "cooling_load"]].mean().reset_index()
    shape_agg_melted = pd.melt(shape_agg, id_vars=["shape_id"], value_vars=["heating_load", "cooling_load"],
                               var_name="Load Type", value_name="Mean Energy Load (kWh/m²)")
    shape_agg_melted["Load Type"] = shape_agg_melted["Load Type"].map({
        "heating_load": "Heating Load",
        "cooling_load": "Cooling Load"
    })
    
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        x="shape_id",
        y="Mean Energy Load (kWh/m²)",
        hue="Load Type",
        data=shape_agg_melted,
        palette={"Heating Load": "#e74c3c", "Cooling Load": "#3498db"},
        edgecolor="black",
        alpha=0.85
    )
    
    plt.title("CoolBuild AI – Mean Energy Loads Across 12 Building Shape Archetypes", fontsize=14, fontweight="bold")
    plt.xlabel("Building Shape ID (Geometry Archetype)", fontsize=11)
    plt.ylabel("Mean Load (kWh/m²)", fontsize=11)
    plt.legend(title="Load Type", frameon=True)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
            
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "07_shape_groups_energy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_orientation_impact_analysis(df: pd.DataFrame) -> str:
    """
    Focused investigation on Orientation (X6).
    Visualizes why orientation has virtually zero correlation with cooling load (r = 0.01).
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df_copy = df.copy()
    orientation_labels = {2: "North", 3: "East", 4: "South", 5: "West"}
    df_copy["orientation_label"] = df_copy["orientation"].map(orientation_labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot of cooling load across orientations
    sns.boxplot(x="orientation_label", y="cooling_load", hue="orientation_label", data=df_copy, ax=axes[0],
                palette="Blues_d", legend=False, boxprops=dict(alpha=0.8))
    axes[0].set_title("Cooling Load by Building Orientation", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Orientation Facing", fontsize=11)
    axes[0].set_ylabel("Cooling Load (kWh/m²)", fontsize=11)
    
    # Calculate orientation means
    means = df_copy.groupby("orientation_label")["cooling_load"].mean()
    for i, (name, val) in enumerate(means.items()):
        axes[0].text(i, val + 1.5, f"Mean: {val:.2f}", ha="center", fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6))
        
    # By height interaction
    sns.pointplot(x="orientation_label", y="cooling_load", hue="overall_height", data=df_copy,
                  ax=axes[1], palette={3.5: "#2ecc71", 7.0: "#9b59b6"}, markers=["o", "s"], capsize=0.1)
    axes[1].set_title("Orientation vs Cooling Load (Split by Building Height)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Orientation Facing", fontsize=11)
    axes[1].set_ylabel("Cooling Load (kWh/m²)", fontsize=11)
    axes[1].legend(title="Overall Height (m)")
    
    plt.suptitle("Orientation Sensitivity Analysis: Demonstrating Negligible Thermal Impact (r = 0.015)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "08_orientation_impact_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def run_full_eda() -> dict:
    """Run complete EDA pipeline and generate all artifacts."""
    print("Executing complete EDA pipeline...")
    df = load_clean_dataset()
    
    desc = compute_descriptive_statistics(df)
    
    plots = {
        "feature_distributions": plot_feature_distributions(df),
        "target_distributions": plot_target_distributions(df),
        "correlation_heatmap": plot_correlation_heatmap(df),
        "feature_vs_cooling_load": plot_feature_vs_cooling_load(df),
        "feature_vs_heating_load": plot_feature_vs_heating_load(df),
        "heating_vs_cooling_coupling": plot_heating_vs_cooling_coupling(df),
        "shape_groups_energy": plot_shape_groups_energy(df),
        "orientation_impact_analysis": plot_orientation_impact_analysis(df),
    }
    print(f"EDA Completed successfully! Generated {len(plots)} visualization plots in {PLOTS_DIR}")
    return plots


if __name__ == "__main__":
    run_full_eda()
