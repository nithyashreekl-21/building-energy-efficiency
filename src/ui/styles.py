"""Design System & Custom CSS for CoolBuild AI.
Implements the Climate-Tech SaaS aesthetic with Deep Forest Green & Sage Palette.
"""

def get_custom_css() -> str:
    """Return the global custom CSS stylesheet for CoolBuild AI."""
    return """
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --color-primary: #174D3A;
        --color-primary-dark: #11382A;
        --color-primary-light: #20634B;
        --color-secondary: #A8C3A0;
        --color-accent: #D5E8B5;
        --color-accent-subtle: #EDF5E0;
        --color-bg: #F7F8F5;
        --color-card: #FFFFFF;
        --color-text: #202820;
        --color-text-muted: #5E6D5E;
        --color-text-light: #8A998A;
        --color-border: #E2E8DE;
        --color-border-subtle: #EEF2EB;
        --color-danger: #D9534F;
        --color-danger-bg: #FDF2F2;
        --color-warning: #C07D2B;
        --color-warning-bg: #FEF8ED;
        --color-success: #174D3A;
        --color-success-bg: #EFF7F2;
        --shadow-sm: 0 1px 3px rgba(23, 77, 58, 0.05);
        --shadow-md: 0 4px 14px rgba(23, 77, 58, 0.08);
        --shadow-lg: 0 10px 25px rgba(23, 77, 58, 0.12);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-full: 9999px;
    }

    /* Global Typography & App Layout */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: var(--color-bg) !important;
        color: var(--color-text) !important;
    }

    /* Main container width & padding */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1380px !important;
    }

    /* Top Header Bar */
    .top-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--color-border);
    }
    .top-header-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--color-primary);
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin: 0;
    }
    .top-header-subtitle {
        font-size: 0.95rem;
        color: var(--color-text-muted);
        margin-top: 0.25rem;
        font-weight: 400;
    }

    /* LEFT SIDEBAR STYLING - Premium Dark Forest Green */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11382A 0%, #174D3A 60%, #1B5742 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #F7F8F5 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFFFFF !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12) !important;
        margin: 1.2rem 0 !important;
    }
    
    /* Sidebar Brand Card */
    .sidebar-brand-box {
        padding: 1.2rem 1rem;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--radius-md);
        margin-bottom: 1.2rem;
        text-align: left;
    }
    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-brand-tagline {
        font-size: 0.8rem;
        color: var(--color-secondary) !important;
        margin-top: 0.35rem;
        line-height: 1.35;
        font-weight: 500;
    }
    .sidebar-version-pill {
        display: inline-block;
        background: rgba(213, 232, 181, 0.2);
        color: var(--color-accent) !important;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: var(--radius-full);
        margin-top: 0.5rem;
        border: 1px solid rgba(213, 232, 181, 0.3);
    }
    .sidebar-status-box {
        margin-top: 0.75rem;
        padding: 0.5rem 0.7rem;
        border-radius: 6px;
        font-size: 0.75rem;
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        line-height: 1.35;
    }
    .sidebar-status-box.provisional {
        background: rgba(254, 243, 199, 0.12);
        border: 1px solid rgba(251, 191, 36, 0.3);
        color: #FDE68A !important;
    }
    .sidebar-status-box.validated {
        background: rgba(213, 232, 181, 0.15);
        border: 1px solid rgba(213, 232, 181, 0.35);
        color: #D5E8B5 !important;
    }
    .sidebar-status-box .status-title {
        font-weight: 700;
    }
    .sidebar-status-box .status-desc {
        opacity: 0.88;
        font-size: 0.72rem;
    }

    /* Sidebar Radio / Navigation styling */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.35rem !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.6rem 0.85rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        color: #E2E8DE !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: var(--color-secondary) !important;
        color: #FFFFFF !important;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
        background: rgba(213, 232, 181, 0.18) !important;
        border-color: var(--color-accent) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    [data-testid="stSidebar"] .stRadio input[type="radio"] {
        accent-color: var(--color-accent) !important;
    }

    /* Sidebar Footer Info Card */
    .sidebar-footer-card {
        padding: 0.9rem;
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
        color: #B4C4B4 !important;
        line-height: 1.4;
        margin-top: 1.5rem;
    }

    /* PREMIUM CARD CONTAINERS */
    .saas-card {
        background-color: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.25rem;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .saas-card:hover {
        border-color: #D1DCCD;
        box-shadow: var(--shadow-md);
    }
    .saas-card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--color-primary);
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .saas-card-body {
        font-size: 0.95rem;
        color: var(--color-text);
        line-height: 1.55;
    }

    /* METRIC TILES */
    .metric-hero-card {
        background: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1.35rem 1.25rem;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-hero-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--color-secondary);
    }
    .metric-hero-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--color-primary) 0%, var(--color-secondary) 100%);
    }
    .metric-hero-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--color-text-muted);
        margin-bottom: 0.4rem;
    }
    .metric-hero-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--color-primary);
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .metric-hero-unit {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--color-text-muted);
        margin-left: 0.25rem;
    }
    .metric-hero-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    .delta-positive {
        color: #174D3A;
    }
    .delta-negative {
        color: #C07D2B;
    }

    /* LARGE ENERGY RESULT CARDS (Cooling vs Heating) */
    .energy-display-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .energy-card-cooling {
        background: linear-gradient(135deg, #FFFFFF 0%, #F3F8F5 100%);
        border: 2px solid var(--color-primary);
        border-radius: var(--radius-lg);
        padding: 1.6rem;
        box-shadow: var(--shadow-md);
        position: relative;
    }
    .energy-card-cooling .badge {
        background: var(--color-accent);
        color: var(--color-primary);
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.25rem 0.6rem;
        border-radius: var(--radius-full);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .energy-card-heating {
        background: linear-gradient(135deg, #FFFFFF 0%, #FDF8F5 100%);
        border: 1px solid #EAD8CE;
        border-radius: var(--radius-lg);
        padding: 1.6rem;
        box-shadow: var(--shadow-sm);
    }
    .energy-card-value {
        font-size: 2.75rem;
        font-weight: 800;
        color: var(--color-primary);
        line-height: 1;
        margin: 0.75rem 0 0.35rem 0;
        letter-spacing: -0.03em;
    }
    .energy-card-heating .energy-card-value {
        color: #A0522D;
    }
    .energy-card-subtitle {
        font-size: 0.88rem;
        color: var(--color-text-muted);
        line-height: 1.4;
    }

    /* HERO BANNER FOR HOME */
    .hero-banner {
        background: linear-gradient(135deg, #11382A 0%, #174D3A 55%, #1F6049 100%);
        border-radius: var(--radius-lg);
        padding: 2.25rem 2rem;
        color: #FFFFFF;
        box-shadow: var(--shadow-lg);
        margin-bottom: 1.75rem;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(213, 232, 181, 0.2);
    }
    @media (max-width: 992px) {
        .hero-banner {
            flex-direction: column;
            align-items: stretch;
            padding: 1.75rem 1.5rem;
        }
    }
    .hero-content {
        flex: 1.35;
        min-width: 280px;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 0.85rem;
        color: #FFFFFF !important;
    }
    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.85rem;
        }
    }
    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.55;
        color: #E2EFE0 !important;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(213, 232, 181, 0.22);
        color: var(--color-accent) !important;
        border: 1px solid rgba(213, 232, 181, 0.4);
        border-radius: var(--radius-full);
        font-size: 0.8rem;
        font-weight: 700;
        padding: 0.3rem 0.85rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .hero-meta {
        font-size: 0.82rem;
        color: var(--color-accent);
        font-weight: 500;
        line-height: 1.4;
    }
    .hero-illustration-container {
        flex: 1;
        min-width: 280px;
        max-width: 460px;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(213, 232, 181, 0.28);
        border-radius: var(--radius-md);
        padding: 1rem;
        backdrop-filter: blur(10px);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    .hero-illustration-caption {
        margin-top: 0.75rem;
        text-align: center;
        width: 100%;
    }
    .hero-caption-title {
        display: block;
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--color-accent);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .hero-caption-sub {
        display: block;
        font-size: 0.72rem;
        color: #C2D6C0;
        margin-top: 0.2rem;
    }

    /* STEP CARDS (How it works) */
    .step-card {
        background: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1.4rem;
        height: 100%;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease, border-color 0.2s ease;
        position: relative;
    }
    .step-card:hover {
        transform: translateY(-3px);
        border-color: var(--color-primary);
        box-shadow: var(--shadow-md);
    }
    .step-number-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background-color: var(--color-primary);
        color: var(--color-accent);
        border-radius: var(--radius-full);
        font-weight: 800;
        font-size: 0.95rem;
        margin-bottom: 0.85rem;
    }
    .step-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--color-primary);
        margin-bottom: 0.4rem;
    }
    .step-desc {
        font-size: 0.875rem;
        color: var(--color-text-muted);
        line-height: 1.5;
    }

    /* BADGES & STATUS PILLS */
    .badge-status {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.75rem;
        border-radius: var(--radius-full);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-status.validated {
        background: #E8F5E9;
        color: #174D3A;
        border: 1px solid #A8C3A0;
    }
    .badge-status.provisional {
        background: #FFF8E1;
        color: #8D6E18;
        border: 1px solid #FFE082;
    }

    /* PARAMETER SUMMARY PILLS */
    .param-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.7rem;
        background: #F0F4EE;
        border: 1px solid var(--color-border);
        border-radius: var(--radius-sm);
        font-size: 0.82rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .param-pill.modified {
        background: #EAF4EE;
        border-color: #A8C3A0;
        color: var(--color-primary);
        font-weight: 700;
    }
    .param-pill.locked {
        background: #F5F5F5;
        border-color: #E0E0E0;
        color: var(--color-text-muted);
    }

    /* BUTTONS */
    .stButton > button {
        border-radius: var(--radius-sm) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--color-primary) !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--color-primary-dark) !important;
        color: var(--color-accent) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: var(--color-primary) !important;
        border: 1px solid var(--color-border) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #F7F8F5 !important;
        border-color: var(--color-primary) !important;
    }

    /* Form Inputs & Sliders */
    .stSlider > div > div > div > div {
        background-color: var(--color-primary) !important;
    }
    .stSelectbox > div > div, 
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--color-border) !important;
        background-color: #FFFFFF !important;
        font-size: 0.95rem !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus {
        border-color: var(--color-primary) !important;
        box-shadow: 0 0 0 2px rgba(23, 77, 58, 0.15) !important;
    }

    /* Tables & Dataframes */
    .stDataFrame, .stTable {
        border-radius: var(--radius-sm) !important;
        overflow: hidden !important;
        border: 1px solid var(--color-border) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem !important;
        border-bottom: 2px solid var(--color-border) !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 0.5rem !important;
        font-weight: 600 !important;
        color: var(--color-text-muted) !important;
        border-bottom: 3px solid transparent !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--color-primary) !important;
        border-bottom-color: var(--color-primary) !important;
        font-weight: 700 !important;
    }

    /* Section Subheaders */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--color-primary);
        margin: 1.5rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
"""
