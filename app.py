"""
EV Adoption in India — Streamlit Dashboard
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Adoption in India",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# THEME / GLOBAL CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Main background ── */
[data-testid="stAppViewContainer"] {
    background: #f4f6f9;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1b2a;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c9d6e3 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e8f0fe !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p { color: #8da9c4 !important; font-size: 0.78rem; }
[data-testid="stSidebar"] hr { border-color: #1e3a5f; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #22a261;
    border-radius: 10px;
    padding: 18px 22px 14px 22px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #0d1b2a !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    line-height: 1.3;
    word-break: break-word;
}

/* ── Section headings ── */
h1 { color: #0d1b2a !important; font-weight: 800; }
h2 { color: #0d1b2a !important; font-weight: 700; margin-top: 0.4rem; }
h3 { color: #1e3a5f !important; font-weight: 600; margin-bottom: 0.25rem; }
h4 { color: #2d5986 !important; font-weight: 600; margin-top: 1.4rem; margin-bottom: 0.2rem; }

/* ── Source / footnote text ── */
.source-note {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    margin-bottom: 1.2rem;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #4b6584 !important;
    padding: 0.5rem 1rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #22a261 !important;
    border-bottom: 3px solid #22a261 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; }

/* ── Dividers ── */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# COLOUR / STYLE CONSTANTS
# ──────────────────────────────────────────────────────────
# Primary green palette for all charts
GREEN_SEQ  = "Greens"          # sequential scale
GREEN_DARK = "#166534"
GREEN_MID  = "#22a261"
GREEN_LITE = "#86efac"
NAVY       = "#0d1b2a"
AMBER      = "#d97706"

# Multi-series palette: green-anchored + complementary
MULTI_PALETTE = [
    "#22a261", "#2563eb", "#d97706", "#7c3aed",
    "#dc2626", "#0891b2", "#65a30d", "#db2777",
    "#0d9488", "#9333ea", "#ea580c", "#1d4ed8",
]

CHART_DEFAULTS = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color="#1e293b"),
    margin=dict(l=10, r=10, t=48, b=10),
    hoverlabel=dict(bgcolor="#ffffff", font_size=12, bordercolor="#e2e8f0"),
)

def apply_chart_style(fig, title="", height=None):
    """Apply consistent layout to any Plotly figure."""
    layout = dict(**CHART_DEFAULTS)
    if title:
        layout["title"] = dict(text=title, font=dict(size=14, color=NAVY, weight="bold"), x=0, xanchor="left")
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=11))
    return fig

def source_note(text: str):
    st.markdown(f'<p class="source-note">Source: {text}</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, filename)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df

def clean_numeric(series: pd.Series) -> pd.Series:
    """Remove commas/quotes and convert to numeric."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace('"', "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0).astype(int)

CATEGORY_LABELS = {
    "FOUR WHEELER (INVALID CARRIAGE)": "4W Invalid Carriage",
    "HEAVY GOODS VEHICLE": "Heavy Goods",
    "HEAVY MOTOR VEHICLE": "Heavy Motor",
    "HEAVY PASSENGER VEHICLE": "Heavy Passenger",
    "LIGHT GOODS VEHICLE": "Light Goods",
    "LIGHT MOTOR VEHICLE": "Light Motor",
    "LIGHT PASSENGER VEHICLE": "Light Passenger",
    "MEDIUM GOODS VEHICLE": "Medium Goods",
    "MEDIUM PASSENGER VEHICLE": "Medium Passenger",
    "MEDIUM MOTOR VEHICLE": "Medium Motor",
    "OTHER THAN MENTIONED ABOVE": "Other",
    "THREE WHEELER(NT)": "3-Wheeler (NT)",
    "TWO WHEELER (INVALID CARRIAGE)": "2W Invalid Carriage",
    "THREE WHEELER(T)": "3-Wheeler (T)",
    "TWO WHEELER(NT)": "2-Wheeler (NT)",
    "TWO WHEELER(T)": "2-Wheeler (T)",
}

CAT_CODE_LABELS = {
    "2W": "Two-Wheeler", "3W": "Three-Wheeler", "4W": "Four-Wheeler",
    "LMV": "Light Motor Vehicle", "HGV": "Heavy Goods", "BUS": "Bus",
    "HPV": "Heavy Passenger", "Constr": "Construction",
}

# ──────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_all_data():
    errors = []

    # ── 1. EV Maker by Place ──────────────────────────────
    try:
        makers_place = load_csv("EV Maker by Place.csv")
        makers_place.columns = ["EV_Maker", "Place", "State"]
        makers_place = makers_place.dropna(subset=["EV_Maker"])
        makers_place["EV_Maker"] = makers_place["EV_Maker"].str.strip().str.strip('"')
        makers_place["State"] = makers_place["State"].str.strip()
        makers_place["Place"] = makers_place["Place"].str.strip()
    except Exception as e:
        errors.append(f"EV Maker by Place.csv: {e}")
        makers_place = pd.DataFrame(columns=["EV_Maker", "Place", "State"])

    # ── 2. EV category monthly data 2001–2024 ─────────────
    try:
        ev_cat = load_csv("ev_cat_01-24.csv")
        ev_cat = ev_cat[ev_cat["Date"].astype(str).str.strip() != "0"]
        ev_cat["Date"] = pd.to_datetime(ev_cat["Date"], format="%d/%m/%y", errors="coerce")
        ev_cat = ev_cat.dropna(subset=["Date"])
        ev_cat["Year"] = ev_cat["Date"].dt.year
        ev_cat["Month"] = ev_cat["Date"].dt.month
        ev_cat["YearMonth"] = ev_cat["Date"].dt.to_period("M").astype(str)
        num_cols = [c for c in ev_cat.columns if c not in ("Date", "Year", "Month", "YearMonth")]
        for col in num_cols:
            ev_cat[col] = clean_numeric(ev_cat[col])
        ev_cat["Total"] = ev_cat[num_cols].sum(axis=1)
    except Exception as e:
        errors.append(f"ev_cat_01-24.csv: {e}")
        ev_cat = pd.DataFrame()

    # ── 3. EV sales by makers & category 2015–2024 ────────
    try:
        sales = load_csv("ev_sales_by_makers_and_cat_15-24.csv")
        sales.columns = [c.strip() for c in sales.columns]
        sales["Maker"] = sales["Maker"].astype(str).str.strip().str.strip('"')
        sales["Cat"] = sales["Cat"].astype(str).str.strip()
        year_cols = [c for c in sales.columns if re.fullmatch(r"\d{4}", c)]
        for col in year_cols:
            sales[col] = clean_numeric(sales[col])
        sales["Total"] = sales[year_cols].sum(axis=1)
    except Exception as e:
        errors.append(f"ev_sales_by_makers_and_cat_15-24.csv: {e}")
        sales = pd.DataFrame()

    # ── 4. Operational charging stations ─────────────────
    try:
        pcs = load_csv("OperationalPC.csv")
        pcs.columns = ["State", "PCS"]
        pcs = pcs.dropna(subset=["State"])
        pcs["State"] = pcs["State"].str.strip()
        pcs["PCS"] = clean_numeric(pcs["PCS"])
    except Exception as e:
        errors.append(f"OperationalPC.csv: {e}")
        pcs = pd.DataFrame(columns=["State", "PCS"])

    # ── 5. Vehicle class total registrations ─────────────
    try:
        veh_class = load_csv("Vehicle Class - All.csv")
        veh_class.columns = ["Vehicle_Class", "Total_Registration"]
        veh_class = veh_class.dropna(subset=["Vehicle_Class"])
        veh_class["Total_Registration"] = clean_numeric(veh_class["Total_Registration"])
        veh_class["Short_Label"] = veh_class["Vehicle_Class"].map(CATEGORY_LABELS).fillna(veh_class["Vehicle_Class"])
    except Exception as e:
        errors.append(f"Vehicle Class - All.csv: {e}")
        veh_class = pd.DataFrame()

    return makers_place, ev_cat, sales, pcs, veh_class, errors


# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
makers_place, ev_cat, sales, pcs, veh_class, load_errors = load_all_data()

if load_errors:
    for err in load_errors:
        st.error(f"⚠️ Data loading issue — {err}")

# ──────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:12px 4px 4px 4px'>"
        "<span style='font-size:1.3rem;font-weight:800;color:#e8f0fe;letter-spacing:-0.01em;'>"
        "⚡ EV India</span><br>"
        "<span style='font-size:0.75rem;color:#8da9c4;'>Interactive Market Dashboard</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:10px 0 14px 0;'>", unsafe_allow_html=True)

    # ── Time range ───────────────────────────────────────
    st.markdown("<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.08em;color:#4a7fa5 !important;margin-bottom:4px;'>TIME RANGE</p>", unsafe_allow_html=True)
    if not ev_cat.empty:
        all_years = sorted(ev_cat["Year"].unique().tolist())
        sel_years = st.select_slider(
            "Year range",
            options=all_years,
            value=(min(all_years), max(all_years)),
            label_visibility="collapsed",
        )
    else:
        sel_years = (2001, 2024)

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

    # ── Category ─────────────────────────────────────────
    st.markdown("<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.08em;color:#4a7fa5 !important;margin-bottom:4px;'>VEHICLE CATEGORY</p>", unsafe_allow_html=True)
    if not sales.empty:
        cat_options = sorted(sales["Cat"].unique().tolist())
        sel_cats = st.multiselect(
            "Category",
            cat_options,
            default=cat_options,
            label_visibility="collapsed",
        )
        if not sel_cats:
            sel_cats = cat_options
    else:
        sel_cats = []

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

    # ── Manufacturer ─────────────────────────────────────
    st.markdown("<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.08em;color:#4a7fa5 !important;margin-bottom:4px;'>MANUFACTURER (TOP 30)</p>", unsafe_allow_html=True)
    if not sales.empty:
        top_makers = (
            sales.groupby("Maker")["Total"].sum()
            .sort_values(ascending=False)
            .head(30)
            .index.tolist()
        )
        sel_makers = st.multiselect(
            "Manufacturer",
            top_makers,
            default=top_makers[:10],
            label_visibility="collapsed",
        )
        if not sel_makers:
            sel_makers = top_makers[:10]
    else:
        sel_makers = []

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

    # ── State ────────────────────────────────────────────
    st.markdown("<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.08em;color:#4a7fa5 !important;margin-bottom:4px;'>STATE (CHARGING)</p>", unsafe_allow_html=True)
    if not pcs.empty:
        state_options = sorted(pcs["State"].unique().tolist())
        sel_states = st.multiselect(
            "State",
            state_options,
            default=state_options,
            label_visibility="collapsed",
        )
        if not sel_states:
            sel_states = state_options
    else:
        sel_states = []

    st.markdown("<hr style='margin:14px 0 8px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.7rem;color:#4a7fa5 !important;'>Data: Vahan Dashboard / EVREPORTER<br>Kaggle: India EV Market Data</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────
# DERIVED DATASETS (after filter)
# ──────────────────────────────────────────────────────────
ev_cat_f = ev_cat[(ev_cat["Year"] >= sel_years[0]) & (ev_cat["Year"] <= sel_years[1])] if not ev_cat.empty else ev_cat
sales_f  = sales[sales["Cat"].isin(sel_cats)] if not sales.empty else sales

year_cols_sales = [c for c in sales.columns if re.fullmatch(r"\d{4}", c)] if not sales.empty else []

# ──────────────────────────────────────────────────────────
# OVERVIEW KPIs
# ──────────────────────────────────────────────────────────
def compute_kpis():
    total_ev = int(ev_cat_f["Total"].sum()) if not ev_cat_f.empty else 0

    lead_maker = "N/A"
    if not sales.empty and year_cols_sales:
        maker_total = sales.groupby("Maker")[year_cols_sales].sum().sum(axis=1)
        lead_maker = maker_total.idxmax() if not maker_total.empty else "N/A"

    lead_cat = "N/A"
    if not veh_class.empty:
        top_idx = veh_class["Total_Registration"].idxmax()
        lead_cat = veh_class.loc[top_idx, "Short_Label"]

    top_pcs_state = "N/A"
    if not pcs.empty:
        top_pcs_state = pcs.loc[pcs["PCS"].idxmax(), "State"]

    return total_ev, lead_maker, lead_cat, top_pcs_state


total_ev, lead_maker, lead_cat, top_pcs_state = compute_kpis()

# ──────────────────────────────────────────────────────────
# TITLE BAR
# ──────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:2px;'>⚡ EV Adoption in India</h1>"
    "<p style='color:#475569;font-size:1rem;margin-top:0;margin-bottom:12px;'>"
    "India EV market trends, manufacturer performance, and charging infrastructure."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr style='margin-bottom:20px;'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# SECTION 1 — OVERVIEW METRICS
# ──────────────────────────────────────────────────────────
def _kpi_card(label: str, value: str, is_numeric: bool = False) -> str:
    """Return an HTML KPI card. Numeric cards use a large bold value;
    text cards use a smaller, fully-visible value that wraps gracefully."""
    value_style = (
        "font-size:1.55rem;font-weight:700;color:#0d1b2a;line-height:1.25;"
        if is_numeric else
        "font-size:0.97rem;font-weight:600;color:#0d1b2a;line-height:1.4;"
        "word-break:break-word;white-space:normal;"
    )
    return (
        f"<div style='background:#ffffff;border:1px solid #e2e8f0;"
        f"border-top:3px solid #22a261;border-radius:10px;"
        f"padding:16px 18px 14px 18px;box-shadow:0 1px 6px rgba(0,0,0,0.07);height:100%;'>"
        f"<p style='margin:0 0 6px 0;font-size:0.72rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.05em;color:#64748b;'>{label}</p>"
        f"<p style='margin:0;{value_style}'>{value}</p>"
        f"</div>"
    )

st.markdown("## 📊 Overview")
c1, c2, c3, c4 = st.columns(4, gap="medium")
c1.markdown(_kpi_card("Total EV Registrations", f"{total_ev:,}", is_numeric=True),  unsafe_allow_html=True)
c2.markdown(_kpi_card("Leading Manufacturer",   lead_maker),                         unsafe_allow_html=True)
c3.markdown(_kpi_card("Top Vehicle Category",   lead_cat),                            unsafe_allow_html=True)
c4.markdown(_kpi_card("Top State · Charging",   top_pcs_state),                       unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Trends",
    "🚗 Categories",
    "🏭 Manufacturers",
    "🔌 Charging",
    "📍 Maker Locations",
    "🔮 Forecast",
    "💡 Recommendations",
])

# ══════════════════════════════════════════════════════════
# TAB 1 — HISTORICAL TRENDS
# ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Historical EV Registration Trends")

    if ev_cat.empty:
        st.warning("Monthly EV data not available.")
    else:
        view = st.radio("View", ["Annual", "Monthly"], horizontal=True, key="trend_view")

        if view == "Annual":
            yearly = ev_cat_f.groupby("Year")["Total"].sum().reset_index()
            fig = px.bar(
                yearly, x="Year", y="Total",
                labels={"Total": "Registrations", "Year": ""},
                color="Total",
                color_continuous_scale=GREEN_SEQ,
                text_auto=True,
            )
            fig.update_traces(textposition="outside", textfont_size=10, marker_line_width=0)
            fig.update_layout(coloraxis_showscale=False, xaxis=dict(tickmode="linear", dtick=2))
            apply_chart_style(fig, title="Annual EV Registrations")
            st.plotly_chart(fig, use_container_width=True)
            source_note("ev_cat_01-24.csv · Vahan Dashboard")

            # YoY growth
            yearly["YoY_Growth_%"] = yearly["Total"].pct_change() * 100
            fig2 = px.line(
                yearly.dropna(subset=["YoY_Growth_%"]),
                x="Year", y="YoY_Growth_%",
                labels={"YoY_Growth_%": "Growth (%)", "Year": ""},
                markers=True,
                color_discrete_sequence=[GREEN_MID],
            )
            fig2.add_hline(y=0, line_dash="dot", line_color="#94a3b8",
                           annotation_text="0 %", annotation_font_size=11)
            apply_chart_style(fig2, title="Year-over-Year Growth Rate (%)")
            st.plotly_chart(fig2, use_container_width=True)

        else:  # Monthly
            monthly = ev_cat_f.groupby("YearMonth")["Total"].sum().reset_index()
            monthly = monthly.sort_values("YearMonth")
            fig = px.line(
                monthly, x="YearMonth", y="Total",
                labels={"Total": "Registrations", "YearMonth": ""},
                color_discrete_sequence=[GREEN_MID],
            )
            fig.update_traces(line_width=2)
            fig.update_layout(xaxis=dict(tickangle=45, nticks=30))
            apply_chart_style(fig, title="Monthly EV Registrations")
            st.plotly_chart(fig, use_container_width=True)
            source_note("ev_cat_01-24.csv · Vahan Dashboard")

        # Heatmap
        cat_cols = [c for c in ev_cat.columns if c not in ("Date", "Year", "Month", "YearMonth", "Total")]
        heatmap_df = ev_cat_f.groupby("Year")[cat_cols].sum()
        heatmap_df.columns = [CATEGORY_LABELS.get(c, c) for c in heatmap_df.columns]
        heatmap_df = heatmap_df.loc[:, heatmap_df.sum() > 0]

        fig3 = px.imshow(
            heatmap_df.T,
            aspect="auto",
            color_continuous_scale="YlGn",
            labels={"x": "Year", "y": "Category", "color": "Registrations"},
        )
        fig3.update_xaxes(tickmode="linear", dtick=2)
        apply_chart_style(fig3, title="Registration Volume — Category × Year Heatmap")
        st.plotly_chart(fig3, use_container_width=True)
        source_note("ev_cat_01-24.csv · Vahan Dashboard")

# ══════════════════════════════════════════════════════════
# TAB 2 — VEHICLE CATEGORIES
# ══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Vehicle Category Analysis")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        if not veh_class.empty:
            vc_plot = veh_class[veh_class["Total_Registration"] > 0].copy()
            fig = px.pie(
                vc_plot, values="Total_Registration", names="Short_Label",
                color_discrete_sequence=MULTI_PALETTE,
                hole=0.45,
            )
            fig.update_traces(
                textposition="inside", textinfo="percent+label",
                textfont_size=11, insidetextorientation="radial",
            )
            apply_chart_style(fig, title="Share by Vehicle Class (All-time)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if not veh_class.empty:
            vc_sorted = vc_plot.sort_values("Total_Registration", ascending=True)
            fig2 = px.bar(
                vc_sorted, x="Total_Registration", y="Short_Label",
                orientation="h",
                labels={"Total_Registration": "Registrations", "Short_Label": ""},
                color="Total_Registration",
                color_continuous_scale=GREEN_SEQ,
                text_auto=True,
            )
            fig2.update_traces(textfont_size=10, marker_line_width=0)
            fig2.update_layout(coloraxis_showscale=False)
            apply_chart_style(fig2, title="Total Registrations by Category")
            st.plotly_chart(fig2, use_container_width=True)

    source_note("Vehicle Class - All.csv · Vahan Dashboard")

    if not ev_cat.empty:
        st.markdown("#### Annual Category Breakdown")
        cat_cols = [c for c in ev_cat.columns if c not in ("Date", "Year", "Month", "YearMonth", "Total")]
        short_labels = [CATEGORY_LABELS.get(c, c) for c in cat_cols]
        label_to_col = dict(zip(short_labels, cat_cols))
        sel_cat_labels = st.multiselect(
            "Select categories to compare",
            short_labels,
            default=[l for l in short_labels if "2-Wheeler" in l or "3-Wheeler" in l],
            key="cat_multisel",
        )
        if sel_cat_labels:
            plot_cols = [label_to_col[l] for l in sel_cat_labels]
            cat_yearly = ev_cat_f.groupby("Year")[plot_cols].sum().reset_index()
            cat_yearly_long = cat_yearly.melt("Year", var_name="Category", value_name="Registrations")
            cat_yearly_long["Category"] = cat_yearly_long["Category"].map(CATEGORY_LABELS).fillna(cat_yearly_long["Category"])
            fig3 = px.line(
                cat_yearly_long, x="Year", y="Registrations", color="Category",
                markers=True,
                color_discrete_sequence=MULTI_PALETTE,
                labels={"Year": "", "Registrations": "Registrations"},
            )
            fig3.update_traces(line_width=2, marker_size=5)
            apply_chart_style(fig3, title="Annual Registrations — Selected Categories")
            st.plotly_chart(fig3, use_container_width=True)
            source_note("ev_cat_01-24.csv · Vahan Dashboard")

    if not sales.empty and year_cols_sales:
        st.markdown("#### Sales by Category (2015–2024)")
        cat_sales = sales_f.groupby("Cat")[year_cols_sales].sum().reset_index()
        cat_sales_long = cat_sales.melt("Cat", var_name="Year", value_name="Sales")
        cat_sales_long["Category"] = cat_sales_long["Cat"].map(CAT_CODE_LABELS).fillna(cat_sales_long["Cat"])
        fig4 = px.bar(
            cat_sales_long, x="Year", y="Sales", color="Category",
            barmode="stack",
            color_discrete_sequence=MULTI_PALETTE,
            labels={"Year": "", "Sales": "Sales"},
        )
        fig4.update_traces(marker_line_width=0)
        apply_chart_style(fig4, title="EV Sales by Category (2015–2024)")
        st.plotly_chart(fig4, use_container_width=True)
        source_note("ev_sales_by_makers_and_cat_15-24.csv · Vahan Dashboard")

# ══════════════════════════════════════════════════════════
# TAB 3 — MANUFACTURERS
# ══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Manufacturer Market Share & Growth")

    if sales.empty or not year_cols_sales:
        st.warning("Sales data not available.")
    else:
        maker_total = (
            sales[sales["Maker"].isin(sel_makers)]
            .groupby("Maker")[year_cols_sales].sum()
        )
        maker_total["Grand_Total"] = maker_total.sum(axis=1)
        maker_total = maker_total.sort_values("Grand_Total", ascending=False)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig = px.pie(
                maker_total.head(15).reset_index(),
                values="Grand_Total", names="Maker",
                color_discrete_sequence=MULTI_PALETTE,
                hole=0.4,
            )
            fig.update_traces(
                textposition="inside", textinfo="percent+label",
                textfont_size=10, insidetextorientation="radial",
            )
            apply_chart_style(fig, title="Market Share — Top 15 Manufacturers (2015–2024)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top10_bar = maker_total.head(10).reset_index()
            fig2 = px.bar(
                top10_bar.sort_values("Grand_Total"),
                x="Grand_Total", y="Maker",
                orientation="h",
                labels={"Grand_Total": "Total Sales", "Maker": ""},
                color="Grand_Total",
                color_continuous_scale=GREEN_SEQ,
                text_auto=True,
            )
            fig2.update_traces(textfont_size=10, marker_line_width=0)
            fig2.update_layout(coloraxis_showscale=False)
            apply_chart_style(fig2, title="Top 10 Manufacturers — Total EV Sales")
            st.plotly_chart(fig2, use_container_width=True)

        source_note("ev_sales_by_makers_and_cat_15-24.csv · Vahan Dashboard")

        st.markdown("#### Annual Sales Growth — Top 10 Manufacturers")
        top_makers_list = maker_total.head(10).index.tolist()
        maker_yearly = (
            sales[sales["Maker"].isin(top_makers_list)]
            .groupby("Maker")[year_cols_sales].sum()
            .reset_index()
        )
        maker_yearly_long = maker_yearly.melt("Maker", var_name="Year", value_name="Sales")

        fig3 = px.line(
            maker_yearly_long, x="Year", y="Sales", color="Maker",
            markers=True,
            color_discrete_sequence=MULTI_PALETTE,
            labels={"Year": "", "Sales": "Sales"},
        )
        fig3.update_traces(line_width=2, marker_size=5)
        apply_chart_style(fig3, title="Annual EV Sales — Top 10 Manufacturers")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Category Mix per Manufacturer")
        cat_maker = (
            sales[sales["Maker"].isin(top_makers_list)]
            .groupby(["Maker", "Cat"])["Total"].sum()
            .reset_index()
        )
        cat_maker["Category"] = cat_maker["Cat"].map(CAT_CODE_LABELS).fillna(cat_maker["Cat"])
        fig4 = px.bar(
            cat_maker, x="Maker", y="Total", color="Category",
            barmode="stack",
            color_discrete_sequence=MULTI_PALETTE,
            labels={"Total": "Sales", "Maker": ""},
        )
        fig4.update_traces(marker_line_width=0)
        fig4.update_layout(xaxis_tickangle=30)
        apply_chart_style(fig4, title="EV Sales by Category — Top 10 Manufacturers")
        st.plotly_chart(fig4, use_container_width=True)
        source_note("ev_sales_by_makers_and_cat_15-24.csv · Vahan Dashboard")

# ══════════════════════════════════════════════════════════
# TAB 4 — CHARGING INFRASTRUCTURE
# ══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Charging Infrastructure Analysis")

    if pcs.empty:
        st.warning("Charging station data not available.")
    else:
        pcs_f = pcs[pcs["State"].isin(sel_states)] if sel_states else pcs

        # ── KPI row ──────────────────────────────────────
        if not ev_cat.empty:
            total_evs_latest = ev_cat[ev_cat["Year"] == ev_cat["Year"].max()]["Total"].sum()
            avg_pcs = pcs_f["PCS"].mean()
            evs_per_pcs = total_evs_latest / max(pcs_f["PCS"].sum(), 1)
            kc1, kc2, kc3 = st.columns(3, gap="medium")
            kc1.metric("Total Charging Stations", f"{pcs_f['PCS'].sum():,}")
            kc2.metric("Avg PCS per State", f"{avg_pcs:,.0f}")
            kc3.metric("Est. EVs per Station", f"{evs_per_pcs:,.0f}")
            st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

        # ── Top 15 bar chart + full ranking side by side ──
        col1, col2 = st.columns([3, 2], gap="large")

        with col1:
            pcs_top15 = pcs_f.nlargest(15, "PCS").sort_values("PCS", ascending=True)
            fig = px.bar(
                pcs_top15, x="PCS", y="State", orientation="h",
                labels={"PCS": "No. of Operational PCS", "State": ""},
                color="PCS",
                color_continuous_scale=GREEN_SEQ,
                text_auto=True,
            )
            fig.update_traces(textfont_size=10, marker_line_width=0)
            fig.update_layout(coloraxis_showscale=False, height=480)
            apply_chart_style(fig, title="Top 15 States by Operational PCS")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top5 = pcs_f.nlargest(5, "PCS")[["State", "PCS"]].reset_index(drop=True)
            bottom5 = pcs_f.nsmallest(5, "PCS")[["State", "PCS"]].reset_index(drop=True)
            st.markdown("**Top 5 States**")
            st.dataframe(top5, use_container_width=True, hide_index=True)
            st.markdown("**Bottom 5 States**")
            st.dataframe(bottom5, use_container_width=True, hide_index=True)

        source_note("OperationalPC.csv · Ministry of Power / EVREPORTER")

        # ── Full state ranking ────────────────────────────
        st.markdown("#### Full State Ranking")
        pcs_all_sorted = pcs_f.sort_values("PCS", ascending=False).reset_index(drop=True)
        pcs_all_sorted.index += 1
        st.dataframe(pcs_all_sorted, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 5 — MAKER LOCATIONS
# ══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### EV Manufacturer Location Analysis")

    if makers_place.empty:
        st.warning("Manufacturer location data not available.")
    else:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            state_count = makers_place["State"].value_counts().reset_index()
            state_count.columns = ["State", "Manufacturer_Count"]
            fig = px.bar(
                state_count.sort_values("Manufacturer_Count", ascending=True),
                x="Manufacturer_Count", y="State",
                orientation="h",
                labels={"Manufacturer_Count": "No. of Manufacturers", "State": ""},
                color="Manufacturer_Count",
                color_continuous_scale=GREEN_SEQ,
                text_auto=True,
            )
            fig.update_traces(textfont_size=10, marker_line_width=0)
            fig.update_layout(coloraxis_showscale=False, height=520)
            apply_chart_style(fig, title="EV Manufacturers by State")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            city_count = makers_place["Place"].value_counts().reset_index()
            city_count.columns = ["City", "Count"]
            fig2 = px.pie(
                city_count.head(12), values="Count", names="City",
                color_discrete_sequence=MULTI_PALETTE,
                hole=0.35,
            )
            fig2.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
            apply_chart_style(fig2, title="Top Manufacturing Cities")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        source_note("EV Maker by Place.csv")

        st.markdown("#### Manufacturer Ecosystem — State → City → Maker")
        fig3 = px.treemap(
            makers_place, path=["State", "Place", "EV_Maker"],
            color="State",
            color_discrete_sequence=MULTI_PALETTE,
        )
        fig3.update_traces(textfont_size=12)
        apply_chart_style(fig3, title="EV Manufacturer Ecosystem")
        fig3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Full Manufacturer Directory")
        st.dataframe(
            makers_place.sort_values("State").reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
        source_note("EV Maker by Place.csv")

# ══════════════════════════════════════════════════════════
# TAB 6 — FORECAST
# ══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### EV Registration Forecast")

    st.info(
        "⚠️ **Forecast Disclaimer:** All projections are **estimates only**, produced by polynomial "
        "regression (degree 2) trained on historical annual registration data. They do not account "
        "for policy shifts, subsidy expiry, supply-chain disruptions, or macroeconomic factors. "
        "Treat values as indicative trends, not predictions.",
        icon=None,
    )

    if ev_cat.empty:
        st.warning("Historical data not available for forecasting.")
    else:
        yearly_total = ev_cat.groupby("Year")["Total"].sum().reset_index()
        yearly_total = yearly_total.sort_values("Year")

        fc_col1, fc_col2 = st.columns(2, gap="large")
        horizon = fc_col1.slider("Forecast horizon (years ahead)", 1, 10, 5)
        train_from = fc_col2.select_slider(
            "Train model from year",
            options=sorted(yearly_total["Year"].tolist()),
            value=2015,
        )

        train_df = yearly_total[yearly_total["Year"] >= train_from]
        X_train = train_df[["Year"]].values
        y_train = train_df["Total"].values

        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X_train, y_train)

        future_years = np.arange(
            yearly_total["Year"].max() + 1,
            yearly_total["Year"].max() + 1 + horizon,
        )
        X_future = future_years.reshape(-1, 1)
        y_pred_future = model.predict(X_future).clip(min=0).astype(int)
        y_pred_train  = model.predict(X_train).clip(min=0).astype(int)

        hist_df = pd.DataFrame({"Year": yearly_total["Year"], "Registrations": yearly_total["Total"], "Type": "Historical"})
        fit_df  = pd.DataFrame({"Year": train_df["Year"], "Registrations": y_pred_train, "Type": "Model Fit"})
        fore_df = pd.DataFrame({"Year": future_years, "Registrations": y_pred_future, "Type": "Forecast (Estimate)"})
        full_df = pd.concat([hist_df, fit_df, fore_df], ignore_index=True)

        color_map = {"Historical": GREEN_MID, "Model Fit": "#2563eb", "Forecast (Estimate)": AMBER}
        dash_map  = {"Historical": "solid",   "Model Fit": "dot",     "Forecast (Estimate)": "dash"}

        fig = go.Figure()
        for dtype, grp in full_df.groupby("Type", sort=False):
            fig.add_trace(go.Scatter(
                x=grp["Year"], y=grp["Registrations"],
                mode="lines+markers",
                name=dtype,
                line=dict(color=color_map[dtype], dash=dash_map[dtype], width=2),
                marker=dict(size=6),
            ))
        fig.add_vrect(
            x0=yearly_total["Year"].max() + 0.5,
            x1=future_years[-1] + 0.5,
            fillcolor="rgba(217,119,6,0.07)",
            line_width=0,
            annotation_text="Forecast zone (estimates)",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color=AMBER,
        )
        apply_chart_style(fig, title="EV Registration Forecast — Polynomial Regression (ESTIMATES)")
        fig.update_layout(
            legend=dict(orientation="h", y=-0.15, font_size=11),
            hovermode="x unified",
            xaxis_title="Year",
            yaxis_title="Registrations",
        )
        st.plotly_chart(fig, use_container_width=True)
        source_note("ev_cat_01-24.csv · Model: polynomial regression (degree 2) on annual totals")

        st.markdown("#### Forecast Values *(Estimates Only)*")
        fore_table = pd.DataFrame({
            "Year": future_years,
            "Estimated Registrations": [f"{v:,}" for v in y_pred_future],
        })
        st.dataframe(fore_table.set_index("Year"), use_container_width=True)

        if not sales.empty and year_cols_sales:
            st.markdown("#### Category-Level Forecast *(Estimates Only)*")
            cat_yearly = sales.groupby("Cat")[year_cols_sales].sum().T.reset_index()
            cat_yearly.columns.name = None
            cat_yearly = cat_yearly.rename(columns={"index": "Year"})
            cat_yearly["Year"] = cat_yearly["Year"].astype(int)

            cat_fore_rows = []
            for cat in cat_yearly.columns[1:]:
                sub = cat_yearly[["Year", cat]].dropna()
                if len(sub) < 3:
                    continue
                Xc = sub[["Year"]].values
                yc = sub[cat].values
                mc = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                mc.fit(Xc, yc)
                yc_pred = mc.predict(X_future).clip(min=0).astype(int)
                for yr, val in zip(future_years, yc_pred):
                    cat_fore_rows.append({"Year": yr, "Category": CAT_CODE_LABELS.get(cat, cat), "Forecast": val})

            if cat_fore_rows:
                cat_fore_df = pd.DataFrame(cat_fore_rows)
                fig2 = px.bar(
                    cat_fore_df, x="Year", y="Forecast", color="Category",
                    barmode="stack",
                    color_discrete_sequence=MULTI_PALETTE,
                    labels={"Year": "", "Forecast": "Estimated Sales"},
                )
                fig2.update_traces(marker_line_width=0)
                apply_chart_style(fig2, title="Forecasted EV Sales by Category — ESTIMATES ONLY")
                st.plotly_chart(fig2, use_container_width=True)
                source_note("ev_sales_by_makers_and_cat_15-24.csv · Estimates based on polynomial regression")

# ══════════════════════════════════════════════════════════
# TAB 7 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### Data-Backed Recommendations")

    if not pcs.empty:
        top3_pcs   = pcs.nlargest(3, "PCS")[["State", "PCS"]].values.tolist()
        low_pcs    = pcs[pcs["PCS"] < 50].sort_values("PCS")
        low_states = low_pcs["State"].tolist()[:5]
        total_pcs_val = pcs["PCS"].sum()

    if not ev_cat.empty:
        yr_max     = int(ev_cat["Year"].max())
        yr_prev    = yr_max - 1
        ev_last    = int(ev_cat[ev_cat["Year"] == yr_max]["Total"].sum())
        ev_prev_v  = int(ev_cat[ev_cat["Year"] == yr_prev]["Total"].sum())
        growth_pct = round((ev_last - ev_prev_v) / max(ev_prev_v, 1) * 100, 1)

    if not sales.empty and year_cols_sales:
        top_maker_name = (
            sales.groupby("Maker")[year_cols_sales].sum()
            .sum(axis=1).idxmax()
        )
        top_cat_code = (
            sales.groupby("Cat")[year_cols_sales].sum()
            .sum(axis=1).idxmax()
        )
        top_cat_name = CAT_CODE_LABELS.get(top_cat_code, top_cat_code)

    insights = []

    if not ev_cat.empty:
        insights.append({
            "icon": "📈",
            "title": "Accelerating Adoption",
            "body": (
                f"EV registrations grew by <strong>{growth_pct}%</strong> in {yr_max} vs {yr_prev}, "
                f"reaching <strong>{ev_last:,}</strong> registrations. The compounding trajectory indicates "
                "strong momentum — infrastructure investment must keep pace."
            ),
        })

    if not sales.empty:
        insights.append({
            "icon": "🏆",
            "title": "Market Concentration Risk",
            "body": (
                f"<strong>{top_maker_name}</strong> dominates the market. A diversified supplier ecosystem "
                "is essential for resilience. Policy should incentivise emerging manufacturers "
                "and reduce single-vendor dependency."
            ),
        })
        insights.append({
            "icon": "🛵",
            "title": f"{top_cat_name}s Lead the Segment",
            "body": (
                f"<strong>{top_cat_name}s</strong> account for the largest share of EV sales. "
                "Subsidies and charging solutions tailored for two- and three-wheelers "
                "will have the highest impact on mass adoption."
            ),
        })

    if not pcs.empty:
        low_states_str = ", ".join(low_states) if low_states else "several states"
        insights.append({
            "icon": "🔌",
            "title": "Charging Infrastructure Gaps",
            "body": (
                f"States like <strong>{low_states_str}</strong> have fewer than 50 public charging stations. "
                f"India's total of <strong>{total_pcs_val:,}</strong> PCS is inadequate for the projected EV fleet. "
                "Targeted investment in under-served states is critical."
            ),
        })
        top3_str = ", ".join([f"{s} ({c:,})" for s, c in top3_pcs])
        insights.append({
            "icon": "🌆",
            "title": "Urban Concentration of Charging",
            "body": (
                f"<strong>{top3_str}</strong> together hold a disproportionate share of charging stations. "
                "Highway charging corridors and Tier-2/3 city coverage must be prioritised "
                "to enable long-distance EV travel across India."
            ),
        })

    if not makers_place.empty:
        mfr_state = makers_place["State"].value_counts().idxmax()
        insights.append({
            "icon": "🏭",
            "title": f"Manufacturing Hub: {mfr_state}",
            "body": (
                f"<strong>{mfr_state}</strong> hosts the highest number of EV manufacturers. "
                "Developing EV manufacturing clusters in other states can create jobs, "
                "reduce logistics costs, and decentralise the supply chain."
            ),
        })

    insights.append({
        "icon": "⚡",
        "title": "Policy Priorities",
        "body": (
            "1. <strong>Extend FAME-III</strong> subsidies with higher allocations for 2W &amp; 3W commercial EVs.<br>"
            "2. <strong>Mandate EV charging</strong> in all new residential and commercial buildings.<br>"
            "3. <strong>Battery-swapping standards</strong> for two-wheelers to lower entry cost.<br>"
            "4. <strong>Green energy integration</strong> — pair charging stations with solar where possible.<br>"
            "5. <strong>Rural EV finance schemes</strong> to accelerate adoption in low-income regions."
        ),
    })

    for insight in insights:
        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                border:1px solid #e2e8f0;
                border-left:4px solid #22a261;
                border-radius:8px;
                padding:16px 20px;
                margin-bottom:12px;">
                <p style="margin:0 0 6px 0;font-size:0.95rem;font-weight:700;color:#0d1b2a;">
                    {insight['icon']}&nbsp; {insight['title']}
                </p>
                <p style="margin:0;font-size:0.875rem;color:#475569;line-height:1.75;">
                    {insight['body']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    source_note("Analysis based on Vahan Dashboard / EVREPORTER data. Forecasts are estimates from polynomial regression on historical registrations.")

# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:32px;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:0.78rem;padding:8px 0 16px 0;'>"
    "EV Adoption in India &nbsp;·&nbsp; Data: Vahan Dashboard / EVREPORTER &nbsp;·&nbsp; "
    "Built with Streamlit &amp; Plotly"
    "</div>",
    unsafe_allow_html=True,
)
