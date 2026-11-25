"""CoolBuild AI – Intelligent Building Energy Optimizer
Modern Climate-Tech SaaS Frontend Application.
Designing Smarter Buildings. Reducing Cooling Energy.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Backend logic imports (PRESERVED 100% UNTOUCHED)
from src.data_loader import (
    load_clean_dataset,
    get_feature_and_target_columns,
    PRETTY_FEATURE_NAMES,
    PRETTY_TARGET_NAMES
)
from src.preprocessing import FEATURE_CONSTRAINTS
from src.train import predict_loads, MODEL_CONFIGS
from src.recommendation import optimize_building_design
from src.community_validation import (
    load_all_validations,
    save_community_validation,
    get_active_changeable_parameters,
    STAKEHOLDER_ROLES
)

# UI Modules
from src.ui.styles import get_custom_css
from src.ui.components import (
    clean_html,
    render_top_header,
    render_sidebar_branding,
    render_metric_card,
    render_hero_banner,
    render_step_card,
    render_energy_display_cards,
    render_stakeholder_banner
)
from src.ui.charts import (
    create_before_after_chart,
    create_feature_importance_chart,
    create_model_comparison_bar_chart,
    create_thermal_gauge_chart,
    PALETTE
)

# Page configuration
st.set_page_config(
    page_title="CoolBuild AI – Intelligent Building Energy Optimizer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global custom SaaS styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize Session State
if "current_building" not in st.session_state:
    st.session_state.current_building = {
        "relative_compactness": 0.76,
        "surface_area": 661.5,
        "wall_area": 416.5,
        "roof_area": 122.5,
        "overall_height": 7.0,
        "orientation": 2,
        "glazing_area": 0.40,
        "glazing_area_distribution": 2
    }

if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"


# Sidebar Navigation & Branding
changeable_params, fixed_params, is_provisional, latest_stakeholder = get_active_changeable_parameters()
render_sidebar_branding(is_provisional, latest_stakeholder)

nav_options = [
    "Home",
    "Building Analysis",
    "Model Performance",
    "Feature Importance",
    "Building Optimizer",
    "Comparison Dashboard",
    "Community Validation"
]

nav_icons = {
    "Home": "🏠",
    "Building Analysis": "🔍",
    "Model Performance": "📊",
    "Feature Importance": "🧠",
    "Building Optimizer": "⚡",
    "Comparison Dashboard": "📈",
    "Community Validation": "🤝"
}

# Sync session state with navigation
def on_nav_change():
    st.session_state.nav_page = st.session_state.sidebar_nav

selected_nav = st.sidebar.radio(
    "Navigation Menu",
    nav_options,
    index=nav_options.index(st.session_state.nav_page) if st.session_state.nav_page in nav_options else 0,
    format_func=lambda x: f"{nav_icons.get(x, '•')} {x}",
    key="sidebar_nav",
    on_change=on_nav_change
)
current_page = st.session_state.nav_page

st.sidebar.markdown(clean_html("""
<div class="sidebar-footer-card">
    <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 0.25rem;">Project Overview</div>
    College Community Engineering Project<br>
    <strong>Dataset:</strong> UCI Energy Efficiency (#242)<br>
    <strong>Target:</strong> Passive Cooling Load Reduction
</div>
"""), unsafe_allow_html=True)


# =========================================================================
# PAGE 1: HOME & OVERVIEW
# =========================================================================
if current_page == "Home":
    render_top_header(
        title="CoolBuild AI",
        subtitle="Designing Smarter Buildings. Reducing Cooling Energy.",
        badge_text="SaaS Climate Platform" if not is_provisional else "Provisional Mode",
        badge_status="validated" if not is_provisional else "provisional"
    )

    # Hero Banner with Sustainable Architectural Illustration
    render_hero_banner()

    # Call-to-Action Action Buttons
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        if st.button("🚀 Analyze Your Building", type="primary", use_container_width=True):
            st.session_state.nav_page = "Building Analysis"
            st.rerun()
    with c_btn2:
        if st.button("📊 Explore Model Performance", type="secondary", use_container_width=True):
            st.session_state.nav_page = "Model Performance"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Authentic Project Metric Cards
    st.markdown('<div class="section-title">Key Platform Benchmarks</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            label="Dataset Size",
            value="768",
            unit="Configurations",
            subtext="Authentic UCI Energy Efficiency repository"
        )
    with m2:
        render_metric_card(
            label="Building Input Features",
            value="8",
            unit="Parameters",
            subtext="Form, geometry, height, orientation & glazing"
        )
    with m3:
        render_metric_card(
            label="Prediction Targets",
            value="2",
            unit="Energy Loads",
            subtext="Heating Load & Cooling Load (kWh/m²)"
        )
    with m4:
        render_metric_card(
            label="ML Algorithms",
            value="4",
            unit="Trained Models",
            subtext="Linear, Decision Tree, Random Forest & SVR"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # "How CoolBuild AI Works" Visual Workflow
    st.markdown('<div class="section-title">How CoolBuild AI Works</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_step_card(
            number=1,
            title="Building Parameters",
            desc="Enter your building's footprint, surface dimensions, orientation, and window glazing characteristics.",
            icon_svg=""
        )
    with s2:
        render_step_card(
            number=2,
            title="ML Prediction",
            desc="Machine learning models predict both heating and cooling energy loads based on physical envelope transfer.",
            icon_svg=""
        )
    with s3:
        render_step_card(
            number=3,
            title="Intelligent Optimization",
            desc="Explore feasible building alternatives, strictly respecting local stakeholder and site constraints.",
            icon_svg=""
        )
    with s4:
        render_step_card(
            number=4,
            title="Energy Insights",
            desc="Compare predicted cooling requirements before and after optimization with plain-language engineering explanations.",
            icon_svg=""
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Overview & Objectives
    st.markdown(clean_html("""
    <div class="saas-card">
        <div class="saas-card-header">
            <span>🎯</span> Project Background & Community Mission
        </div>
        <div class="saas-card-body">
            <p>
                Space cooling in residential buildings is one of the primary drivers of escalating electric grid peak loads, 
                household utility costs, and greenhouse gas emissions. In many developing communities, buildings are constructed 
                without thermal performance analysis, resulting in excessive solar heat gains and thermal discomfort.
            </p>
            <p>
                <strong>CoolBuild AI</strong> solves this by bridging scientific machine learning with on-site masonry and architectural realities. 
                Rather than treating building optimization as a theoretical exercise, the system requires local stakeholders (masons, architects, housing officers) 
                to confirm which building parameters can realistically be modified on site, delivering actionable, zero-cost or low-cost passive cooling interventions.
            </p>
        </div>
    </div>
    """), unsafe_allow_html=True)


# =========================================================================
# PAGE 2: BUILDING LOAD ANALYSIS
# =========================================================================
elif current_page == "Building Analysis":
    render_top_header(
        title="Building Energy Load Analysis",
        subtitle="Specify architectural envelope parameters to predict heating and cooling energy demands.",
        badge_text="Prediction Studio"
    )

    cb = st.session_state.current_building

    # Archetype Presets Selector
    presets = {
        "Custom Configuration": None,
        "Preset 1: Two-Story High-Glazing Villa (High Cooling Demand)": {
            "relative_compactness": 0.76, "surface_area": 661.5, "wall_area": 416.5, "roof_area": 122.5,
            "overall_height": 7.0, "orientation": 2, "glazing_area": 0.40, "glazing_area_distribution": 2
        },
        "Preset 2: Single-Story Low-Glazing Cottage (Moderate Climate)": {
            "relative_compactness": 0.74, "surface_area": 686.0, "wall_area": 245.0, "roof_area": 220.5,
            "overall_height": 3.5, "orientation": 3, "glazing_area": 0.10, "glazing_area_distribution": 1
        },
        "Preset 3: Highly Compact Sustainable Envelope": {
            "relative_compactness": 0.98, "surface_area": 514.5, "wall_area": 294.0, "roof_area": 110.25,
            "overall_height": 7.0, "orientation": 4, "glazing_area": 0.25, "glazing_area_distribution": 3
        }
    }

    selected_preset = st.selectbox("📂 Load Pre-configured Building Archetype (Optional)", list(presets.keys()))
    if presets[selected_preset] is not None:
        st.session_state.current_building = presets[selected_preset]
        cb = st.session_state.current_building

    # 3 Organized Input Sections
    st.markdown('<div class="section-title">1. Building Geometry & Envelope</div>', unsafe_allow_html=True)
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        rc = st.slider(
            "Relative Compactness",
            min_value=0.62, max_value=0.98, value=float(cb["relative_compactness"]), step=0.01,
            help="Ratio of building volume to surface area relative to a reference cube (0.62 = spread out, 0.98 = compact)."
        )
        height = st.radio(
            "Overall Height",
            options=[3.5, 7.0],
            index=0 if float(cb["overall_height"]) == 3.5 else 1,
            format_func=lambda x: f"{x} m ({'Single-Story' if x==3.5 else 'Two-Story'})",
            horizontal=True,
            help="Building vertical height (single vs two-story)."
        )
    with g_col2:
        sa = st.slider(
            "Surface Area (m²)",
            min_value=514.5, max_value=808.5, value=float(cb["surface_area"]), step=12.25,
            help="Total exterior building envelope area."
        )
        wa = st.slider(
            "Wall Area (m²)",
            min_value=245.0, max_value=416.5, value=float(cb["wall_area"]), step=12.25,
            help="Total vertical wall area."
        )
    with g_col3:
        ra = st.slider(
            "Roof Area (m²)",
            min_value=110.25, max_value=220.5, value=float(cb["roof_area"]), step=12.25,
            help="Total horizontal roof area."
        )
        # Physical law sanity check
        geom_law = abs(sa - (2 * ra + wa))
        if geom_law > 0.01:
            st.caption(f"ℹ Envelope Check: Surface ({sa:.1f}) vs 2×Roof + Wall ({2*ra+wa:.1f})")

    st.markdown('<div class="section-title">2. Building Orientation & Compass Facing</div>', unsafe_allow_html=True)
    ori_col1, ori_col2 = st.columns([1, 2])
    with ori_col1:
        ori_map = {2: "North (2)", 3: "East (3)", 4: "South (4)", 5: "West (5)"}
        ori = st.selectbox(
            "Orientation (Primary Solar Facing)",
            options=[2, 3, 4, 5],
            index=[2, 3, 4, 5].index(int(cb["orientation"])),
            format_func=lambda x: ori_map[x],
            help="Primary solar orientation of the main facade."
        )
    with ori_col2:
        st.markdown(clean_html("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8DE; border-radius: 8px; padding: 0.85rem 1rem; font-size: 0.85rem; color: #5E6D5E; margin-top: 1.5rem;">
            <strong>Orientation Sensitivity Note:</strong> Annual simulation results demonstrate that rotating building orientation yields 
            nearly identical cumulative annual cooling load (&plusmn;0.02 kWh/m²). It influences daily peak timing, but not annual kWh demand.
        </div>
        """), unsafe_allow_html=True)

    st.markdown('<div class="section-title">3. Fenestration & Glazing Configuration</div>', unsafe_allow_html=True)
    fz_col1, fz_col2 = st.columns(2)
    with fz_col1:
        glazing_area_map = {0.0: "0% (Opaque Envelope / No Windows)", 0.10: "10% Glazing Area", 0.25: "25% Glazing Area", 0.40: "40% Glazing Area (High)"}
        g_area = st.selectbox(
            "Glazing Area (Window-to-Wall Ratio)",
            options=[0.0, 0.10, 0.25, 0.40],
            index=[0.0, 0.10, 0.25, 0.40].index(float(cb["glazing_area"])),
            format_func=lambda x: glazing_area_map[x],
            help="Proportion of vertical exterior wall area covered by glass windows."
        )
    with fz_col2:
        dist_map = {0: "0: None / Uniform", 1: "1: Uniform Distribution", 2: "2: North-Facing", 3: "3: East-Facing", 4: "4: South-Facing", 5: "5: West-Facing"}
        g_dist = st.selectbox(
            "Glazing Distribution Pattern",
            options=[0, 1, 2, 3, 4, 5],
            index=[0, 1, 2, 3, 4, 5].index(int(cb["glazing_area_distribution"])),
            format_func=lambda x: dist_map[x],
            help="Directional window placement pattern."
        )

    # Save to session state
    st.session_state.current_building = {
        "relative_compactness": rc,
        "surface_area": sa,
        "wall_area": wa,
        "roof_area": ra,
        "overall_height": height,
        "orientation": ori,
        "glazing_area": g_area,
        "glazing_area_distribution": g_dist
    }

    # Model Selector
    algo_col, btn_col = st.columns([1.5, 1])
    with algo_col:
        algo_choice = st.selectbox(
            "Machine Learning Prediction Engine",
            options=["random_forest", "decision_tree", "linear_regression", "svr"],
            format_func=lambda x: MODEL_CONFIGS[x]["name"] + (" ⭐ (Production Model)" if x=="random_forest" else "")
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Predict Energy Loads", type="primary", use_container_width=True):
        preds = predict_loads(st.session_state.current_building, algorithm=algo_choice)
        st.session_state.last_prediction = preds
        st.success("Energy loads successfully computed via trained machine learning model!")

    if "last_prediction" in st.session_state:
        preds = st.session_state.last_prediction
        st.markdown('<div class="section-title">Model Energy Predictions</div>', unsafe_allow_html=True)
        render_energy_display_cards(preds["cooling_load"], preds["heating_load"])

        # Thermal Gauge Indicator
        st.markdown("**Cooling Load Intensity Bar**")
        st.altair_chart(create_thermal_gauge_chart(preds["cooling_load"]), use_container_width=False)
        st.caption("Cooling load benchmark scale: 0 kWh/m² (Zero Energy Building) to 48 kWh/m² (High Solar Influx).")

        # Quick link to Optimizer
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Proceed to Building Optimizer with this Configuration", type="secondary", use_container_width=True):
            st.session_state.nav_page = "Building Optimizer"
            st.rerun()


# =========================================================================
# PAGE 3: MODEL PERFORMANCE & CROSS-VALIDATION
# =========================================================================
elif current_page == "Model Performance":
    render_top_header(
        title="Model Performance & Scientific Benchmarks",
        subtitle="Empirical evaluation of 4 regression algorithms under Random and Grouped Cross-Validation.",
        badge_text="Rigorous Evaluation"
    )

    tab_cv, tab_charts, tab_holdout = st.tabs([
        "🔬 Random CV vs Grouped CV",
        "📊 Comparative Visualizations",
        "📋 Holdout Split Metrics"
    ])

    with tab_cv:
        st.markdown(clean_html("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span>🛡</span> Why Grouped Cross-Validation is the Hallmark of Rigorous ML
            </div>
            <div class="saas-card-body">
                <p>
                    A critical pitfall in spatial and architectural machine learning is <strong>geometric data leakage</strong>:
                </p>
                <ul>
                    <li>The UCI Energy Efficiency dataset contains 768 rows derived from <strong>12 elementary building geometric shapes</strong>.</li>
                    <li>In <strong>Random 5-Fold CV</strong>, configurations of the same shape leak into both training and testing folds (~80% in train, ~20% in test). The model merely interpolates between orientations of shapes it has already memorized, producing an optimistic <strong>R² &ge; 0.969</strong>.</li>
                    <li>In <strong>Grouped 5-Fold CV</strong> (using <code>GroupKFold</code> on the 12 shape clusters), entire geometric shapes are held out strictly in test folds. The model is forced to extrapolate to <strong>unseen architectural forms</strong>, exposing the true generalization boundary.</li>
                </ul>
            </div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown("#### Cooling Load ($Y_2$) Cross-Validation Results")
        csv_cool = os.path.join(PROJECT_ROOT, "results", "evaluation", "cv_comparison_cooling_load.csv")
        if os.path.exists(csv_cool):
            df_cool = pd.read_csv(csv_cool)
            st.dataframe(df_cool, use_container_width=True)

        st.markdown("#### Heating Load ($Y_1$) Cross-Validation Results")
        csv_heat = os.path.join(PROJECT_ROOT, "results", "evaluation", "cv_comparison_heating_load.csv")
        if os.path.exists(csv_heat):
            df_heat = pd.read_csv(csv_heat)
            st.dataframe(df_heat, use_container_width=True)

        # Scientific Summary Box
        st.info(
            r"""
            **Takeaways for Faculty & Reviewers**:
            - **Random Forest** delivers the best overall performance under Random CV ($R^2 = 0.969 \pm 0.006$, $\text{MAE} = 1.032\text{ kWh/m²}$), surpassing the project target of $R^2 \ge 0.95$ and $\text{MAE} \le 1.5\text{ kWh/m²}$.
            - On unseen shapes (Grouped CV), error increases to $5.19\text{ kWh/m²}$, but still beats the mean predictor baseline ($8.81\text{ kWh/m²}$) by $41\%$.
            - Linear Regression generalizes with moderate stability ($R^2 = 0.5815$) because linear hyperplanes extrapolate smoothly, while Decision Trees and SVR with RBF kernels degrade on distant unseen geometry.
            """
        )

    with tab_charts:
        st.markdown('<div class="section-title">Interactive Model Benchmark Charts</div>', unsafe_allow_html=True)
        if os.path.exists(csv_cool):
            c_chart1 = create_model_comparison_bar_chart(df_cool, metric="R2")
            st.altair_chart(c_chart1, use_container_width=True)

            c_chart2 = create_model_comparison_bar_chart(df_cool, metric="MAE")
            st.altair_chart(c_chart2, use_container_width=True)

        st.markdown('<div class="section-title">Publication Figures (300 DPI)</div>', unsafe_allow_html=True)
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            st.image(os.path.join(PROJECT_ROOT, "results", "plots", "09_cv_cooling_load_r2_comparison.png"),
                     caption="Cooling Load: R² Comparison vs Project Target (0.95)")
            st.image(os.path.join(PROJECT_ROOT, "results", "plots", "11_cv_fold_variance_boxplots.png"),
                     caption="Cooling Load: 5-Fold MAE Spread Across Validation Protocols")
        with img_c2:
            st.image(os.path.join(PROJECT_ROOT, "results", "plots", "10_cv_cooling_load_mae_comparison.png"),
                     caption="Cooling Load: MAE vs Target (1.5) & Mean Baseline (8.64)")
            st.image(os.path.join(PROJECT_ROOT, "results", "plots", "09_cv_heating_load_r2_comparison.png"),
                     caption="Heating Load: Cross-Validation Generalization Scores")

    with tab_holdout:
        st.markdown('<div class="section-title">Holdout Test Set Performance (80/20 Partition)</div>', unsafe_allow_html=True)
        summary_json_path = os.path.join(PROJECT_ROOT, "models", "training_summary.json")
        if os.path.exists(summary_json_path):
            with open(summary_json_path, "r", encoding="utf-8") as f:
                ts = json.load(f)
            rows = []
            for tgt, t_data in ts["models"].items():
                for a_key, a_data in t_data["algorithms"].items():
                    m = a_data["metrics"]
                    rows.append({
                        "Energy Target": "Cooling Load (Y2)" if "cooling" in tgt else "Heating Load (Y1)",
                        "Algorithm": a_data["name"],
                        "Train R²": f"{m['train_r2']:.4f}",
                        "Test R²": f"{m['test_r2']:.4f}",
                        "Test MAE (kWh/m²)": f"{m['test_mae']:.3f}",
                        "Test RMSE (kWh/m²)": f"{m['test_rmse']:.3f}"
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown(clean_html("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span>📖</span> Understanding Evaluation Metrics in Beginner-Friendly Terms
            </div>
            <div class="saas-card-body">
                <ul>
                    <li><strong>R² Score (Coefficient of Determination)</strong>: Ranges from 0 to 1. A score of 0.969 means our machine learning model explains <strong>96.9% of the variance</strong> in cooling energy loads.</li>
                    <li><strong>MAE (Mean Absolute Error)</strong>: The average magnitude of prediction error in actual physical units ($kWh/m²$). An MAE of 1.03 means our predictions are typically within ~1.0 kWh/m² of ground-truth simulation.</li>
                    <li><strong>RMSE (Root Mean Squared Error)</strong>: Similar to MAE, but penalizes large outlier errors more severely. A low RMSE indicates that the model rarely makes extreme prediction errors.</li>
                </ul>
            </div>
        </div>
        """), unsafe_allow_html=True)


# =========================================================================
# PAGE 4: FEATURE IMPORTANCE & INTERPRETABILITY
# =========================================================================
elif current_page == "Feature Importance":
    render_top_header(
        title="Feature Importance & Sensitivity Analysis",
        subtitle="Identify which building parameters dictate heating and cooling energy demands.",
        badge_text="Interpretability"
    )

    csv_imp = os.path.join(PROJECT_ROOT, "results", "evaluation", "feature_importance_summary.csv")
    if os.path.exists(csv_imp):
        df_imp = pd.read_csv(csv_imp)

        # Interactive Altair Charts for Cooling & Heating
        st.markdown('<div class="section-title">Interactive Parameter Influence Charts</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.altair_chart(create_feature_importance_chart(df_imp, target="cooling_load"), use_container_width=True)
        with col_c2:
            st.altair_chart(create_feature_importance_chart(df_imp, target="heating_load"), use_container_width=True)

        # Ranked Summary Table
        st.markdown('<div class="section-title">Parameter Influence Ranking Table</div>', unsafe_allow_html=True)
        disp_df = df_imp[["pretty_name", "cooling_mdi", "cooling_perm", "heating_mdi", "heating_perm"]].copy()
        disp_df.columns = ["Building Parameter", "Cooling MDI (%)", "Cooling Permutation (ΔR²)", "Heating MDI (%)", "Heating Permutation (ΔR²)"]
        disp_df["Cooling MDI (%)"] = (disp_df["Cooling MDI (%)"] * 100).map("{:.1f}%".format)
        disp_df["Heating MDI (%)"] = (disp_df["Heating MDI (%)"] * 100).map("{:.1f}%".format)
        disp_df["Cooling Permutation (ΔR²)"] = disp_df["Cooling Permutation (ΔR²)"].map("{:.4f}".format)
        disp_df["Heating Permutation (ΔR²)"] = disp_df["Heating Permutation (ΔR²)"].map("{:.4f}".format)
        st.dataframe(disp_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Orientation Paradox & Causation Clarification
    st.markdown('<div class="section-title">Critical Architectural & ML Insights</div>', unsafe_allow_html=True)
    in_col1, in_col2 = st.columns(2)

    with in_col1:
        st.markdown(clean_html("""
        <div class="saas-card" style="height: 100%;">
            <div class="saas-card-header" style="color: #174D3A;">
                <span>🧭</span> The "Orientation Paradox" Demystified
            </div>
            <div class="saas-card-body">
                <p>
                    <strong>Why does building orientation show almost zero importance (< 1%) for Cooling Load?</strong>
                </p>
                <p>
                    The dataset records cumulative <em>annual</em> cooling energy demand ($kWh/m²$). In symmetrical or four-sided residential envelopes, 
                    rotating the building by 90° or 180° simply alters the time of day at which solar radiation penetrates (morning east facade vs afternoon west facade).
                </p>
                <p>
                    <strong>Practical Rule</strong>: The recommendation engine will <em>never</em> advise rotating an existing building plot to reduce annual cooling demand. 
                    Interventions must focus on window glazing area, shading distribution, and envelope compactness.
                </p>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with in_col2:
        st.markdown(clean_html("""
        <div class="saas-card" style="height: 100%;">
            <div class="saas-card-header" style="color: #C07D2B;">
                <span>⚖</span> Correlation vs. Physical Causation
            </div>
            <div class="saas-card-body">
                <p>
                    <strong>Machine Learning Feature Importance &ne; Direct Physical Causation:</strong>
                </p>
                <ul>
                    <li>In this dataset, single-story buildings (3.5 m) have double the roof area for an equivalent footprint, so "Roof Area" partially captures building height and compactness.</li>
                    <li>Parameters are physically coupled: one cannot arbitrarily increase wall area without altering surface area or building footprint.</li>
                    <li>The CoolBuild AI Recommendation Engine accounts for physical geometry conservation ($\text{Surface Area} = 2 \times \text{Roof Area} + \text{Wall Area}$) to prevent unbuildable designs.</li>
                </ul>
            </div>
        </div>
        """), unsafe_allow_html=True)


# =========================================================================
# PAGE 5: BUILDING DESIGN OPTIMIZER
# =========================================================================
elif current_page == "Building Optimizer":
    render_top_header(
        title="Discover Your Building's Energy-Efficient Potential",
        subtitle="Explore alternative building configurations and identify opportunities to reduce predicted cooling requirements.",
        badge_text="Constrained Optimizer"
    )

    # Active Stakeholder Feasibility Status Banner
    render_stakeholder_banner(is_provisional, latest_stakeholder)

    features, _ = get_feature_and_target_columns()
    cb = st.session_state.current_building

    # Two Main Panels: Left (Current Building) vs Right (Optimization Settings)
    p_left, p_right = st.columns([1, 1.2])

    with p_left:
        st.markdown('<div class="section-title">Current Building Baseline</div>', unsafe_allow_html=True)
        base_preds = predict_loads(cb)
        param_rows = "".join([
            f"<div style='display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #F0F4EE; font-size:0.86rem; color:#202820;'><span>{PRETTY_FEATURE_NAMES.get(f, f)}</span><strong style='font-family:monospace;'>{cb[f]}</strong></div>"
            for f in features
        ])
        baseline_card_html = f"""
        <div class="saas-card">
            <div class="saas-card-header">📐 Active Design Parameters</div>
            <div class="saas-card-body">
                {param_rows}
                <div style="margin-top: 1rem; padding: 0.85rem; background: #F7F8F5; border-radius: 8px; border: 1px solid #E2E8DE;">
                    <div style="font-size: 0.8rem; color: #5E6D5E; font-weight: 700; text-transform: uppercase;">Baseline Cooling Load</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #174D3A;">{base_preds['cooling_load']:.2f} <span style="font-size: 0.9rem;">kWh/m²</span></div>
                    <div style="font-size: 0.8rem; color: #8A998A;">Coupled Heating Load: {base_preds['heating_load']:.2f} kWh/m²</div>
                </div>
            </div>
        </div>
        """
        st.markdown(clean_html(baseline_card_html), unsafe_allow_html=True)

    with p_right:
        st.markdown('<div class="section-title">Optimization Constraints & Settings</div>', unsafe_allow_html=True)
        st.write("Select which parameters are **practically changeable** on your construction site:")

        opt_grid = st.columns(2)
        selected_changeable = []
        for i, feat in enumerate(features):
            with opt_grid[i % 2]:
                is_def = feat in changeable_params
                checked = st.checkbox(PRETTY_FEATURE_NAMES.get(feat, feat), value=is_def, key=f"opt_sel_{feat}")
                if checked:
                    selected_changeable.append(feat)

        locked_list = [PRETTY_FEATURE_NAMES.get(f, f) for f in features if f not in selected_changeable]
        st.caption(f"🔒 Locked Immutable Parameters: {locked_list}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Find Better Building Design", type="primary", use_container_width=True):
            if not selected_changeable:
                st.error("Please mark at least one changeable parameter to explore alternatives.")
            else:
                with st.spinner("Evaluating physically constrained architectural alternatives..."):
                    opt_res = optimize_building_design(cb, changeable_params=selected_changeable)
                    st.session_state.optimization_result = opt_res
                st.success("Optimization search completed successfully!")

    # Optimization Result Display Section
    if st.session_state.optimization_result is not None:
        res = st.session_state.optimization_result
        st.markdown("---")
        st.markdown('<div class="section-title">Optimization Findings & Recommended Alternative</div>', unsafe_allow_html=True)

        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        with res_c1:
            render_metric_card("Original Cooling Load", f"{res['original_loads']['cooling_load']:.2f}", unit="kWh/m²")
        with res_c2:
            render_metric_card("Recommended Cooling Load", f"{res['recommended_loads']['cooling_load']:.2f}", unit="kWh/m²",
                               delta=f"-{res['cooling_reduction']:.2f} kWh/m²", is_positive=True)
        with res_c3:
            render_metric_card("Potential Reduction", f"{res['cooling_reduction_pct']:.1f}%", unit="Savings", is_positive=True)
        with res_c4:
            render_metric_card("Candidates Evaluated", f"{res['total_candidates_evaluated']}", unit="Archetypes", subtext="Physically valid")

        st.markdown("<br>", unsafe_allow_html=True)

        # Before-and-After Altair Chart
        st.altair_chart(
            create_before_after_chart(
                orig_cooling=res["original_loads"]["cooling_load"],
                rec_cooling=res["recommended_loads"]["cooling_load"],
                orig_heating=res["original_loads"]["heating_load"],
                rec_heating=res["recommended_loads"]["heating_load"]
            ),
            use_container_width=True
        )

        # Plain-Language Architectural Rationale
        rationale_html = f"""
        <div class="saas-card">
            <div class="saas-card-header">💡 Practical Engineering Explanation</div>
            <div class="saas-card-body" style="background: #F0F7F2; padding: 1.1rem; border-radius: 8px; border-left: 4px solid #174D3A; color: #174D3A; font-weight: 500; font-size: 0.95rem; line-height: 1.6;">
                {res['engineering_explanation']}
            </div>
        </div>
        """
        st.markdown(clean_html(rationale_html), unsafe_allow_html=True)

        # Modified Parameters Breakdown
        st.markdown("#### Modified Parameters Summary")
        mods = res.get("modified_parameters", {})
        if mods:
            m_rows = []
            for f, d in mods.items():
                m_rows.append({
                    "Parameter": d["parameter_name"],
                    "Original Value": d["original_value"],
                    "Recommended Value": d["recommended_value"],
                    "Difference": f"{d['change']:+}"
                })
            st.dataframe(pd.DataFrame(m_rows), use_container_width=True)
        else:
            st.write("Baseline design is already optimal within the allowed constraints.")

        # Top 5 Evaluated Alternatives Table
        st.markdown("#### Top 5 Feasible Alternatives Evaluated")
        top_list = []
        for i, cand in enumerate(res.get("top_alternatives", [])):
            top_list.append({
                "Rank": f"#{i+1}",
                "Cooling Load (kWh/m²)": cand["cooling_load"],
                "Heating Load (kWh/m²)": cand["heating_load"],
                "Reduction (kWh/m²)": cand["reduction"],
                "Glazing Area": cand["config"]["glazing_area"],
                "Glazing Dist": cand["config"]["glazing_area_distribution"],
                "Compactness": cand["config"]["relative_compactness"],
                "Height": cand["config"]["overall_height"]
            })
        st.dataframe(pd.DataFrame(top_list), use_container_width=True)

        # Jump to Comparison Dashboard button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📈 Open Full Side-by-Side Comparison Dashboard", type="secondary", use_container_width=True):
            st.session_state.nav_page = "Comparison Dashboard"
            st.rerun()


# =========================================================================
# PAGE 6: COMPARISON DASHBOARD
# =========================================================================
elif current_page == "Comparison Dashboard":
    render_top_header(
        title="Before-vs-After Comparison Dashboard",
        subtitle="Comprehensive visual and parametric audit of original vs. recommended building design.",
        badge_text="Executive Summary"
    )

    if st.session_state.optimization_result is None:
        st.info("No active optimization result found. Please run an optimization on **Building Optimizer** first.")
    else:
        res = st.session_state.optimization_result

        # Side-by-Side Design Cards
        card_col1, card_col2 = st.columns(2)

        with card_col1:
            st.markdown(clean_html(f"""
            <div class="saas-card" style="border-left: 5px solid #94A394;">
                <div class="saas-card-header" style="color: #5E6D5E;">
                    <span>🏛</span> ORIGINAL BUILDING DESIGN
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #202820;">
                    {res['original_loads']['cooling_load']:.2f} <span style="font-size: 1rem; font-weight: 600; color: #5E6D5E;">kWh/m² Cooling</span>
                </div>
                <div style="font-size: 0.95rem; color: #5E6D5E; margin-top: 0.25rem;">
                    Heating Load: {res['original_loads']['heating_load']:.2f} kWh/m²
                </div>
                <hr style="margin: 1rem 0; border-color: #E2E8DE;">
                <div style="font-size: 0.85rem; line-height: 1.6;">
                    <strong>Relative Compactness:</strong> {res['original_config']['relative_compactness']}<br>
                    <strong>Surface Area:</strong> {res['original_config']['surface_area']} m²<br>
                    <strong>Wall Area:</strong> {res['original_config']['wall_area']} m²<br>
                    <strong>Roof Area:</strong> {res['original_config']['roof_area']} m²<br>
                    <strong>Overall Height:</strong> {res['original_config']['overall_height']} m<br>
                    <strong>Glazing Area:</strong> {int(res['original_config']['glazing_area']*100)}%<br>
                    <strong>Glazing Distribution:</strong> Pattern {int(res['original_config']['glazing_area_distribution'])}
                </div>
            </div>
            """), unsafe_allow_html=True)

        with card_col2:
            st.markdown(clean_html(f"""
            <div class="saas-card" style="border-left: 5px solid #174D3A; background: #F8FAF7;">
                <div class="saas-card-header" style="color: #174D3A;">
                    <span>🌿</span> RECOMMENDED OPTIMIZED DESIGN
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #174D3A;">
                    {res['recommended_loads']['cooling_load']:.2f} <span style="font-size: 1rem; font-weight: 600; color: #174D3A;">kWh/m² Cooling</span>
                </div>
                <div style="font-size: 0.95rem; color: #174D3A; font-weight: 600; margin-top: 0.25rem;">
                    Potential Reduction: {res['cooling_reduction']:.2f} kWh/m² ({res['cooling_reduction_pct']:.1f}% Savings)
                </div>
                <hr style="margin: 1rem 0; border-color: #A8C3A0;">
                <div style="font-size: 0.85rem; line-height: 1.6;">
                    <strong>Relative Compactness:</strong> {res['recommended_config']['relative_compactness']}<br>
                    <strong>Surface Area:</strong> {res['recommended_config']['surface_area']} m²<br>
                    <strong>Wall Area:</strong> {res['recommended_config']['wall_area']} m²<br>
                    <strong>Roof Area:</strong> {res['recommended_config']['roof_area']} m²<br>
                    <strong>Overall Height:</strong> {res['recommended_config']['overall_height']} m<br>
                    <strong>Glazing Area:</strong> {int(res['recommended_config']['glazing_area']*100)}%<br>
                    <strong>Glazing Distribution:</strong> Pattern {int(res['recommended_config']['glazing_area_distribution'])}
                </div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive Chart
        st.altair_chart(
            create_before_after_chart(
                orig_cooling=res["original_loads"]["cooling_load"],
                rec_cooling=res["recommended_loads"]["cooling_load"],
                orig_heating=res["original_loads"]["heating_load"],
                rec_heating=res["recommended_loads"]["heating_load"]
            ),
            use_container_width=True
        )

        # Full Parameter Audit Table
        st.markdown('<div class="section-title">Side-by-Side Parameter Audit</div>', unsafe_allow_html=True)
        features, _ = get_feature_and_target_columns()
        comp_rows = []
        for f in features:
            p_name = PRETTY_FEATURE_NAMES.get(f, f)
            o_val = res["original_config"][f]
            r_val = res["recommended_config"][f]
            is_mod = f in res.get("modified_parameters", {})
            comp_rows.append({
                "Parameter": p_name,
                "Original Building": o_val,
                "Recommended Alternative": r_val,
                "Status": "🔄 Modified" if is_mod else "🔒 Fixed / Immutable",
                "Difference": f"{r_val - o_val:+}" if is_mod else "0.0"
            })

        comp_rows.append({
            "Parameter": "⭐ Cooling Energy Load (kWh/m²)",
            "Original Building": res["original_loads"]["cooling_load"],
            "Recommended Alternative": res["recommended_loads"]["cooling_load"],
            "Status": "📉 Reduced (Target)",
            "Difference": f"-{res['cooling_reduction']:.2f} ({res['cooling_reduction_pct']:.1f}%)"
        })
        comp_rows.append({
            "Parameter": "Heating Energy Load (kWh/m²)",
            "Original Building": res["original_loads"]["heating_load"],
            "Recommended Alternative": res["recommended_loads"]["heating_load"],
            "Status": "Coupled Demand",
            "Difference": f"{res['heating_difference']:+.2f}"
        })

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        st.caption("⚠ Disclaimer: All energy savings shown are model-predicted potential thermal load reductions based on the UCI Energy Efficiency dataset. Actual electrical energy and billing savings depend on HVAC efficiency (COP/SEER), regional climate conditions, and local electricity tariffs.")


# =========================================================================
# PAGE 7: COMMUNITY VALIDATION
# =========================================================================
elif current_page == "Community Validation":
    render_top_header(
        title="Grounding AI Recommendations in Real-World Construction",
        subtitle="Record local stakeholder surveys to ensure all ML recommendations reflect on-site construction realities.",
        badge_text="Stakeholder Engine"
    )

    st.markdown(clean_html("""
    <div class="saas-card">
        <div class="saas-card-header">
            <span>🤝</span> The Human-in-the-Loop Community Mandate
        </div>
        <div class="saas-card-body">
            <p>
                In practical construction, machine learning algorithms frequently propose mathematically optimal modifications 
                (such as altering building height, foundation footprint, or orientation) that are impossible on site due 
                to plot boundaries, budget constraints, or soil bearing capacity.
            </p>
            <p>
                By recording a local mason, architect, or municipal housing officer's confirmation, the CoolBuild AI engine 
                strictly locks unchangeable parameters and only searches across <strong>locally actionable modifications</strong>.
            </p>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Record New Stakeholder Feasibility Survey</div>', unsafe_allow_html=True)

    with st.form("stakeholder_form_redesign"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Stakeholder Name", placeholder="e.g. Ramesh Patel, Er. Ananya Sharma")
            role = st.selectbox("Stakeholder Role / Profession", STAKEHOLDER_ROLES)
            location = st.text_input("Community Site Location", placeholder="e.g. Ward 4 Community Housing Site, Bengaluru")
            stage = st.selectbox("Project Construction Stage", [
                "Pre-Construction / Site Planning",
                "Masonry & Foundation Underway",
                "Roofing Completed / Fenestration Stage",
                "Retrofit of Existing Building"
            ])

        with col2:
            features, _ = get_feature_and_target_columns()
            changeable_selected = st.multiselect(
                "Parameters that CAN be modified on this site:",
                options=features,
                default=["glazing_area", "glazing_area_distribution"],
                format_func=lambda x: PRETTY_FEATURE_NAMES.get(x, x),
                help="Only select parameters that local builders confirm are physically and financially feasible to adjust."
            )
            notes = st.text_area(
                "Stakeholder Field Observations / Constraints",
                placeholder="e.g. Plot dimensions fix building footprint and orientation. Window shading frames can be altered without structural impact.",
                height=125
            )

        submit = st.form_submit_button("💾 Save Stakeholder Validation Record", type="primary")
        if submit:
            rec = save_community_validation(
                stakeholder_name=name,
                stakeholder_role=role,
                location=location,
                changeable_parameters=changeable_selected,
                observations=notes,
                project_stage=stage
            )
            st.success(f"Stakeholder validation successfully saved (ID: {rec['id']})! The optimizer will now prioritize these parameters.")
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-title">Historical Community Validation Audit Log</div>', unsafe_allow_html=True)
    validations = load_all_validations()
    if validations:
        v_rows = []
        for v in reversed(validations):
            v_rows.append({
                "Date": v.get("timestamp", "")[:10],
                "Stakeholder": v.get("stakeholder_name", "Anonymous"),
                "Role": v.get("stakeholder_role", "Unknown"),
                "Location": v.get("location", ""),
                "Changeable Parameters": ", ".join([PRETTY_FEATURE_NAMES.get(p, p) for p in v.get("changeable_parameters", [])]),
                "Observations": v.get("observations", "")
            })
        st.dataframe(pd.DataFrame(v_rows), use_container_width=True)
    else:
        st.write("No stakeholder validation records logged yet.")
