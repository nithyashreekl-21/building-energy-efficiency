"""Reusable UI Components for CoolBuild AI.
Modern Climate-Tech SaaS Component System with Robust HTML Sanitization.
"""

import streamlit as st
from typing import Dict, Any, Optional, List


def clean_html(html_str: str) -> str:
    """Normalize an HTML or SVG string for safe Streamlit markdown rendering.
    
    Strips leading and trailing whitespace from every line and removes empty lines.
    This strictly guarantees that CommonMark/Python-Markdown never interprets
    indented lines (>= 4 spaces) or blank-line boundaries as raw code blocks (<pre><code>).
    """
    if not html_str:
        return ""
    lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    return "\n".join(lines)


def render_top_header(
    title: str,
    subtitle: str,
    badge_text: Optional[str] = None,
    badge_status: str = "validated"
):
    """Render a clean top header with title, subtitle, and optional live status badge."""
    badge_html = ""
    if badge_text:
        badge_class = "validated" if badge_status == "validated" else "provisional"
        badge_icon = "✔" if badge_status == "validated" else "⚠"
        badge_html = f'<div class="badge-status {badge_class}">{badge_icon} {badge_text}</div>'

    html = f"""
    <div class="top-header-container">
        <div>
            <h1 class="top-header-title">{title}</h1>
            <div class="top-header-subtitle">{subtitle}</div>
        </div>
        {badge_html}
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_sidebar_branding(is_provisional: bool, latest_stakeholder: Optional[Dict[str, Any]] = None):
    """Render sidebar logo, tagline, and community status pill."""
    if is_provisional or latest_stakeholder is None:
        status_html = """
        <div class="sidebar-status-box provisional">
            <span class="status-title">⚠ Feasibility: Provisional</span>
            <span class="status-desc">No field stakeholder logged. Using conservative defaults.</span>
        </div>
        """
    else:
        role = latest_stakeholder.get("stakeholder_role", "").split("/")[0].strip()
        loc = latest_stakeholder.get("location", "Local Site")
        status_html = f"""
        <div class="sidebar-status-box validated">
            <span class="status-title">✔ Validated by {role}</span>
            <span class="status-desc">Site: {loc}</span>
        </div>
        """

    brand_html = f"""
    <div class="sidebar-brand-box">
        <div class="sidebar-brand-title">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D5E8B5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 21h18"/>
                <path d="M5 21V7l8-4v18"/>
                <path d="M19 21V11l-6-4"/>
                <path d="M9 9h1"/>
                <path d="M9 13h1"/>
                <path d="M9 17h1"/>
            </svg>
            <span>CoolBuild AI</span>
        </div>
        <div class="sidebar-brand-tagline">
            Designing Smarter Buildings.<br>Reducing Cooling Energy.
        </div>
        <div class="sidebar-version-pill">CLIMATE-TECH SAAS v2.0</div>
        {status_html}
    </div>
    """
    st.sidebar.markdown(clean_html(brand_html), unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    unit: str = "",
    delta: str = "",
    is_positive: bool = True,
    subtext: str = ""
):
    """Render a single SaaS-style metric card."""
    delta_html = ""
    if delta:
        delta_class = "delta-positive" if is_positive else "delta-negative"
        arrow = "↓" if is_positive else "↑"
        delta_html = f'<div class="metric-hero-delta {delta_class}"><span>{arrow} {delta}</span></div>'

    subtext_html = f'<div style="font-size: 0.78rem; color: #8A998A; margin-top: 0.35rem;">{subtext}</div>' if subtext else ""

    html = f"""
    <div class="metric-hero-card">
        <div class="metric-hero-label">{label}</div>
        <div>
            <span class="metric-hero-value">{value}</span>
            <span class="metric-hero-unit">{unit}</span>
        </div>
        {delta_html}
        {subtext_html}
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_hero_banner():
    """Render the homepage hero banner with an integrated architectural vector visual."""
    hero_html = """
    <div class="hero-banner">
        <div class="hero-content">
            <div class="hero-badge">🌿 AI FOR SUSTAINABLE ARCHITECTURE</div>
            <h1 class="hero-title">Design Buildings That Need Less Cooling.</h1>
            <p class="hero-subtitle">
                An intelligent building energy optimization platform powered by machine learning. 
                Predict thermal loads and discover practically feasible, community-grounded architectural 
                configurations that drastically lower cooling energy demands.
            </p>
            <div class="hero-meta">
                Benchmarked on authentic UCI Energy Efficiency data • 12 Geometry Archetypes • Constrained Local Feasibility
            </div>
        </div>
        <div class="hero-illustration-container">
            <svg viewBox="0 0 460 300" width="100%" height="auto" style="max-height: 240px; display: block;" xmlns="http://www.w3.org/2000/svg">
                <!-- Sun & Solar Radiation Rays -->
                <circle cx="380" cy="50" r="36" fill="#F4D06F" opacity="0.18"/>
                <circle cx="380" cy="50" r="24" fill="#F4D06F" opacity="0.95"/>
                <line x1="380" y1="14" x2="380" y2="4" stroke="#F4D06F" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="412" y1="24" x2="420" y2="16" stroke="#F4D06F" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="348" y1="24" x2="340" y2="16" stroke="#F4D06F" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="345" y1="65" x2="295" y2="95" stroke="#F4D06F" stroke-width="2" stroke-dasharray="4 3"/>
                <text x="310" y="64" fill="#F4D06F" font-size="9" font-weight="700" font-family="sans-serif">Summer Solar Influx</text>
                
                <!-- Ground / Foundation Grade -->
                <rect x="25" y="248" width="410" height="7" rx="3.5" fill="#A8C3A0"/>
                <line x1="12" y1="255" x2="448" y2="255" stroke="#D5E8B5" stroke-width="2" stroke-linecap="round"/>
                <line x1="40" y1="255" x2="30" y2="265" stroke="#A8C3A0" stroke-width="1.5"/>
                <line x1="80" y1="255" x2="70" y2="265" stroke="#A8C3A0" stroke-width="1.5"/>
                <line x1="120" y1="255" x2="110" y2="265" stroke="#A8C3A0" stroke-width="1.5"/>
                <line x1="340" y1="255" x2="330" y2="265" stroke="#A8C3A0" stroke-width="1.5"/>
                <line x1="380" y1="255" x2="370" y2="265" stroke="#A8C3A0" stroke-width="1.5"/>
                <line x1="420" y1="255" x2="410" y2="265" stroke="#A8C3A0" stroke-width="1.5"/>
                
                <!-- Main Building Envelope (Passive High-Performance Archetype) -->
                <rect x="85" y="95" width="205" height="153" rx="4" fill="#FFFFFF" stroke="#D5E8B5" stroke-width="2"/>
                <rect x="290" y="145" width="115" height="103" rx="4" fill="#F7F8F5" stroke="#D5E8B5" stroke-width="2"/>
                
                <!-- Overhanging Cantilevered Shading Roofs (Blocks High Summer Radiation) -->
                <polygon points="70,95 305,95 285,76 90,76" fill="#174D3A" stroke="#D5E8B5" stroke-width="1.5"/>
                <rect x="100" y="71" width="165" height="5" rx="2" fill="#A8C3A0"/>
                <text x="118" y="68" fill="#D5E8B5" font-size="8" font-weight="700" font-family="sans-serif">GREEN ROOF THERMAL MASS</text>
                
                <polygon points="278,145 415,145 398,131 295,131" fill="#174D3A" stroke="#D5E8B5" stroke-width="1.5"/>
                
                <!-- Windows with Passive Shading Louvers / Fins -->
                <rect x="110" y="115" width="55" height="38" rx="3" fill="#EDF5E0" stroke="#174D3A" stroke-width="1.5"/>
                <line x1="115" y1="148" x2="145" y2="120" stroke="#FFFFFF" stroke-width="1.5" opacity="0.6"/>
                <rect x="104" y="110" width="67" height="5" rx="2" fill="#D5E8B5"/>
                
                <rect x="195" y="115" width="55" height="38" rx="3" fill="#EDF5E0" stroke="#174D3A" stroke-width="1.5"/>
                <line x1="200" y1="148" x2="230" y2="120" stroke="#FFFFFF" stroke-width="1.5" opacity="0.6"/>
                <rect x="189" y="110" width="67" height="5" rx="2" fill="#D5E8B5"/>
                
                <rect x="110" y="180" width="55" height="42" rx="3" fill="#EDF5E0" stroke="#174D3A" stroke-width="1.5"/>
                <rect x="104" y="175" width="67" height="5" rx="2" fill="#D5E8B5"/>
                
                <rect x="325" y="165" width="55" height="42" rx="3" fill="#EDF5E0" stroke="#174D3A" stroke-width="1.5"/>
                <rect x="319" y="160" width="67" height="5" rx="2" fill="#D5E8B5"/>
                
                <!-- Main Entrance Airlock -->
                <rect x="200" y="178" width="45" height="70" rx="3" fill="#A8C3A0" stroke="#174D3A" stroke-width="1.5"/>
                <rect x="210" y="188" width="25" height="25" rx="2" fill="#FFFFFF"/>
                <circle cx="238" cy="218" r="2.5" fill="#174D3A"/>
                
                <!-- Passive Natural Cross-Ventilation Streamlines -->
                <path d="M 40 205 Q 75 195 106 200" fill="none" stroke="#D5E8B5" stroke-width="2.2" stroke-dasharray="4 3"/>
                <polygon points="110,200 102,195 103,204" fill="#D5E8B5"/>
                <text x="35" y="192" fill="#D5E8B5" font-size="8.5" font-weight="700" font-family="sans-serif">Cool Air Inflow</text>
                
                <path d="M 140 180 Q 155 150 170 115" fill="none" stroke="#D5E8B5" stroke-width="1.8" stroke-dasharray="3 3"/>
                <path d="M 252 118 Q 275 102 296 92" fill="none" stroke="#D5E8B5" stroke-width="2" stroke-dasharray="4 3"/>
                <polygon points="300,90 291,89 295,98" fill="#D5E8B5"/>
                <text x="250" y="85" fill="#D5E8B5" font-size="8.5" font-weight="700" font-family="sans-serif">Warm Air Exhaust</text>
                
                <!-- Surrounding Vegetation Buffer (Evapotranspirative Microclimate Cooling) -->
                <path d="M 45 248 Q 52 215 58 248" fill="#174D3A"/>
                <circle cx="52" cy="218" r="16" fill="#A8C3A0" opacity="0.9"/>
                <circle cx="52" cy="214" r="11" fill="#D5E8B5" opacity="0.85"/>
                
                <rect x="420" y="235" width="6" height="13" fill="#174D3A"/>
                <circle cx="423" cy="222" r="18" fill="#A8C3A0" opacity="0.9"/>
                <circle cx="423" cy="217" r="13" fill="#D5E8B5" opacity="0.85"/>
                
                <!-- Dimension Marker -->
                <line x1="75" y1="95" x2="75" y2="248" stroke="#A8C3A0" stroke-width="1" stroke-dasharray="2 2"/>
                <text x="46" y="175" fill="#A8C3A0" font-size="8" font-family="monospace">H=7.0m</text>
            </svg>
            <div class="hero-illustration-caption">
                <span class="hero-caption-title">PASSIVE THERMAL ENVELOPE ARCHETYPE</span>
                <span class="hero-caption-sub">Shading Fins • Optimized Glazing Ratio • Cross-Ventilation</span>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(hero_html), unsafe_allow_html=True)


def render_step_card(number: int, title: str, desc: str, icon_svg: str = ""):
    """Render a step card for the 'How it works' section."""
    html = f"""
    <div class="step-card">
        <div class="step-number-badge">0{number}</div>
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_energy_display_cards(
    cooling_load: float,
    heating_load: float,
    cooling_baseline_avg: float = 24.59,
    heating_baseline_avg: float = 22.31
):
    """Render the dual primary energy cards for Cooling Load and Heating Load."""
    cool_diff = cooling_load - cooling_baseline_avg
    heat_diff = heating_load - heating_baseline_avg
    
    cool_status = f"{cool_diff:+.2f} vs dataset average (24.59)"
    heat_status = f"{heat_diff:+.2f} vs dataset average (22.31)"

    cool_color = "#174D3A" if cool_diff <= 0 else "#C07D2B"

    html = f"""
    <div class="energy-display-container">
        <div class="energy-card-cooling">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #174D3A; text-transform: uppercase; letter-spacing: 0.05em;">
                    Target to Reduce
                </span>
                <span class="badge">COOLING LOAD</span>
            </div>
            <div class="energy-card-value">
                {cooling_load:.2f} <span style="font-size: 1.1rem; font-weight: 600; color: #5E6D5E;">kWh/m²</span>
            </div>
            <div class="energy-card-subtitle">
                Model-predicted energy demand to keep indoor spaces at thermal comfort during warm seasons.
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.8rem; font-weight: 600; color: {cool_color};">
                {cool_status}
            </div>
        </div>
        <div class="energy-card-heating">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #A0522D; text-transform: uppercase; letter-spacing: 0.05em;">
                    Coupled Thermal Load
                </span>
                <span style="background: #FDF2EB; color: #A0522D; font-weight: 700; font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 9999px; text-transform: uppercase;">HEATING LOAD</span>
            </div>
            <div class="energy-card-value">
                {heating_load:.2f} <span style="font-size: 1.1rem; font-weight: 600; color: #5E6D5E;">kWh/m²</span>
            </div>
            <div class="energy-card-subtitle">
                Model-predicted energy required to maintain indoor heating comfort during cold seasons.
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.8rem; font-weight: 600; color: #7F8C8D;">
                {heat_status}
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_stakeholder_banner(is_provisional: bool, latest_stakeholder: Optional[Dict[str, Any]] = None):
    """Render a prominent stakeholder status banner."""
    if is_provisional or latest_stakeholder is None:
        html = """
        <div style="background-color: #FEF9EE; border-left: 4px solid #C07D2B; padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 700; color: #8D5B1B; font-size: 0.95rem;">
                <span>⚠</span> PROVISIONAL RECOMMENDATION MODE
            </div>
            <div style="font-size: 0.85rem; color: #6D5025; margin-top: 0.25rem;">
                No on-site stakeholder validation is currently on record. The optimizer is enforcing conservative defaults 
                (glazing proportion and shading distribution only). Visit <strong>Community Validation</strong> to confirm site feasibility.
            </div>
        </div>
        """
    else:
        role = latest_stakeholder.get("stakeholder_role", "Stakeholder")
        name = latest_stakeholder.get("stakeholder_name", "Representative")
        loc = latest_stakeholder.get("location", "Project Site")
        html = f"""
        <div style="background-color: #F0F7F2; border-left: 4px solid #174D3A; padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 700; color: #174D3A; font-size: 0.95rem;">
                <span>✔</span> GROUNDED IN SITE FEASIBILITY: VALIDATED BY {name.upper()} ({role.upper()})
            </div>
            <div style="font-size: 0.85rem; color: #345842; margin-top: 0.25rem;">
                Location: <strong>{loc}</strong>. All recommendations strictly obey approved parameter modifiability.
            </div>
        </div>
        """
    st.markdown(clean_html(html), unsafe_allow_html=True)
