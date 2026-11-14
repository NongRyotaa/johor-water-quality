"""JIE42403 Scientific Visualisation – Johor river water quality 2020 (Streamlit app)."""
import numpy as np
import pandas as pd
import streamlit as st

import charts as C

st.set_page_config(page_title="Johor River Water Quality 2020", page_icon="💧", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1250px;}
      .obj-box {background:#eef6fb; border-left:6px solid #1f77b4; padding:.8rem 1rem; border-radius:6px;
                margin-bottom:.8rem; font-size:1.02rem;}
      .interp {background:#fafafa; border:1px solid #e3e3e3; padding:.9rem 1.1rem; border-radius:8px;}
      div[data-testid="stMetricValue"] {font-size:1.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    return C.load_data()


DATA = get_data()


# --------------------------------------------------------------------------------------
# Sidebar filters (apply to every page)
# --------------------------------------------------------------------------------------
def sidebar_filters() -> pd.DataFrame:
    with st.sidebar:
        st.divider()
        st.markdown("### 🔎 Filters")
        basins = st.multiselect("River basin", C.BASIN_ORDER, default=C.BASIN_ORDER,
                                help="Muar and Skudai have many stations; Johor and Linggi have one each.")
        m_lo, m_hi = st.select_slider("Sampling months", options=C.MONTHS, value=("Jan", "Dec"))
        st.divider()
        st.caption("**Data:** Mohd Yusoff, Z., Abu Bakar, A.A. & Abu Bakar, Z. (2024). *Dataset on Water Quality "
                   "Monitoring Parameters from Various Monitoring Stations in Johor, Malaysia.* Mendeley Data, V1. "
                   "doi:10.17632/929mtp3fyr.1")
        st.caption("Values reported as “< detection limit” are set to half the limit. "
                   "*Conductivity header in the source says mS/cm, but the magnitudes (median ≈ 110) match µS/cm.")
    lo, hi = C.MONTHS.index(m_lo) + 1, C.MONTHS.index(m_hi) + 1
    f = DATA[DATA["basin"].isin(basins) & DATA["month"].between(lo, hi)]
    if f.empty:
        st.warning("No samples match the current filters – widen the basin or month selection.")
        st.stop()
    return f


def header(title, objective):
    st.title(title)
    st.markdown(f'<div class="obj-box"><b>Objective:</b> {objective}</div>', unsafe_allow_html=True)


def summary_box(items):
    """items = list of (label, value, help)."""
    with st.container(border=True):
        st.markdown("**📦 Summary box**")
        cols = st.columns(len(items))
        for col, (label, value, hlp) in zip(cols, items):
            col.metric(label, value, help=hlp)


def interpretation(text):
    st.markdown("#### 📝 Interpretation")
    st.markdown(f'<div class="interp">{text}</div>', unsafe_allow_html=True)
    st.caption("Written for the full dataset (all basins, Jan–Dec). Charts and summary boxes update with the filters.")


# --------------------------------------------------------------------------------------
# Page 1 – Spatial
# --------------------------------------------------------------------------------------
def page_spatial():
    df = sidebar_filters()
    header("Objective 1 · Spatial patterns and pollution hotspots",
           "To map how river water quality varies across the 39 monitoring stations and four basins of Johor, "
           "and to identify the stations and basins that act as pollution hotspots.")
    param = st.selectbox("Parameter to explore", list(C.PARAMS), format_func=C.PARAMS.get, index=1)

    st_tab = C.station_table(df, param)
    top = st_tab.loc[st_tab["mean"].idxmax()]
    by_basin = df.groupby("basin", observed=True)[param].mean().dropna()
    over = (df["NH3N"] > C.NWQS_III["NH3N"]).mean() * 100
    summary_box([
        ("Stations · samples", f"{df['station_id'].nunique()} · {len(df)}", "Monitoring stations and samples in the current filter"),
        (f"Hotspot station ({C.SHORT[param]})", f"{top['station_id']}", f"Highest station mean = {top['mean']:.2f}. {top['location']}"),
        (f"Highest basin mean", f"{by_basin.idxmax()} · {by_basin.max():.2f}", f"Basin with the highest mean {C.PARAMS[param]}"),
        ("NH3-N above Class III", f"{over:.0f}% of samples", "Share of samples with NH3-N > 0.9 mg/L (NWQS Class III limit)"),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(C.fig_station_map(df, param), width="stretch")
    log = c2.toggle("Log scale (box plot)", value=param in ("NH3N", "COND", "Cl", "PO4"))
    c2.plotly_chart(C.fig_basin_box(df, param, log), width="stretch")
    n = st.slider("Stations in ranking", 5, 15, 10)
    st.plotly_chart(C.fig_station_rank(df, param, n), width="stretch")

    interpretation(
        "<b>Skudai is the clear ammonia hotspot.</b> Its stations average 3.3 mg/L NH3-N against about 0.2 mg/L in "
        "Muar and Johor and 0.01 mg/L in Linggi, and 67% of Skudai samples exceed the NWQS Class III limit of 0.9 mg/L "
        "(only 2% of Muar samples do). Seven of the ten highest stations are Skudai stations around Kulai, Senai and "
        "Taman Perling (3JSKU010 = 8.1 mg/L); Sepang Loi (3JMUA022, 2.1 mg/L) is the only Muar station whose mean exceeds the limit. "
        "Skudai also has far higher phosphate (mean 1.27 vs 0.12 mg/L in Muar). Ammonia and phosphate together point "
        "to sewage and industrial discharge in the dense Johor Bahru–Kulai corridor rather than natural causes. "
        "<br><b>Muar, Johor and Linggi have a different problem.</b> Their suspended solids and turbidity are higher "
        "than Skudai's (mean SS 87–234 vs 51 mg/L) while ammonia is low, which indicates sediment from erosion, land "
        "clearing and agriculture rather than wastewater. The box plot also shows that Skudai's range is very wide "
        "(roughly 0.02–13 mg/L), so its stations are not equally affected. "
        "Johor and Linggi have only one station (6 samples) each, so their basin results are indicative only."
    )

# --------------------------------------------------------------------------------------
# Page 2 – Temporal
# --------------------------------------------------------------------------------------
def page_temporal():
    df = sidebar_filters()
    header("Objective 2 · Seasonal variation through 2020",
           "To examine how suspended solids, turbidity, ammonia and iron changed through the year and to find "
           "periods of elevated runoff or pollution.")
    c0, c1, c2_ = st.columns(3)
    p_line = c0.selectbox("Line plot parameter", list(C.PARAMS), format_func=C.PARAMS.get, index=4)
    p_heat = c1.selectbox("Heatmap parameter", list(C.PARAMS), format_func=C.PARAMS.get, index=1)
    p_time = c2_.selectbox("Timeline parameter", list(C.PARAMS), format_func=C.PARAMS.get, index=0)

    param = p_line
    mm = df.groupby("month")[param].mean()
    peak, low = mm.idxmax(), mm.idxmin()
    mx = df.loc[df[param].idxmax()]
    summary_box([
        (f"Peak month ({C.SHORT[param]})", f"{C.MONTHS[peak-1]} · {mm.max():.1f}", "Month with the highest all-station mean"),
        ("Lowest month", f"{C.MONTHS[low-1]} · {mm.min():.1f}", "Month with the lowest all-station mean"),
        ("Peak ÷ lowest", f"{mm.max()/mm.min():.1f}×", "Seasonal amplitude of the monthly mean"),
        ("Highest single reading", f"{mx[param]:.1f}", f"{mx['location']} on {mx['date']:%d %b %Y}"),
        ("Months sampled", f"{df['month'].nunique()} of 12", "No samples exist for April 2020"),
    ])

    st.plotly_chart(C.fig_monthly_lines(df, p_line), width="stretch")
    st.caption("March contains only 12 samples (9 Skudai, 3 Muar) – read that point with care. "
               "No samples were collected in April.")
    c2, c3 = st.columns(2)
    c2.plotly_chart(C.fig_round_heatmap(df, p_heat), width="stretch")
    log = c3.toggle("Log scale (timeline)", value=False)
    c3.plotly_chart(C.fig_timeline(df, p_time, log), width="stretch")

    interpretation(
        "<b>Turbidity and suspended solids peak at the end of the year.</b> The all-station mean turbidity rises from "
        "34 NTU in January to 230 NTU in November (2.3× the annual mean of 101 NTU). In Muar the November mean is 394 "
        "NTU, against 20–146 NTU in the other months, and the four highest turbidity readings (487–677 NTU) were all "
        "taken on 20–22 November at Muar-basin stations such as Rompin and Kampung Londah. This is consistent with "
        "heavy rain at the start of the north-east monsoon washing sediment into the river. Suspended solids show "
        "the same November surge, plus an earlier May peak; 29 of 234 samples (12%) exceed the 150 mg/L Class III "
        "limit, 23 of them in Muar, and the maximum was 754 mg/L at Jambatan Gantung Jelai on 6 June. "
        "<br><b>Ammonia follows a different pattern.</b> In the heatmap it is high only in the Skudai row: "
        "6.1 mg/L in March (round 2), 3.2–3.6 mg/L in rounds 1, 3 and 4, and about 1.5 mg/L in September–December, "
        "while Muar stays at or below 0.5 mg/L and Linggi near zero. The fall in the second half may reflect more "
        "dilution in the wetter months, but one year of data cannot prove the cause. "
        "<br><b>Caution:</b> March was sampled at only 12 stations (9 Skudai, 3 Muar), so its high all-station "
        "NH3-N mean reflects which stations were visited, not a basin-wide March peak. Johor and Linggi cells in the "
        "heatmap are single measurements, and no samples were collected in April."
    )

# --------------------------------------------------------------------------------------
# Page 3 – Relationships and anomalies
# --------------------------------------------------------------------------------------
def page_relationships():
    df = sidebar_filters()
    header("Objective 3 · Relationships between parameters and anomalies",
           "To quantify how the physical and chemical parameters relate to each other and to detect anomalous "
           "readings, such as saline intrusion, that break the normal pattern.")
    method = st.radio("Correlation method", ["spearman", "pearson"], horizontal=True,
                      help="Spearman (rank-based) is more reliable here because the data are strongly skewed.")
    corr = C.corr_matrix(df, method).where(~np.tril(np.ones((len(C.CORR_VARS),) * 2, bool)))
    pair = corr.stack().abs().idxmax()
    tur_ss = C.corr_matrix(df, method).loc["TUR", "SS"]
    saline = df[df["Cl"] > 250]
    summary_box([
        ("Strongest pair", f"{C.SHORT[pair[0]]}–{C.SHORT[pair[1]]}", f"{method.title()} = {C.corr_matrix(df, method).loc[pair]:.2f}"),
        ("Turbidity–SS", f"{tur_ss:.2f}", f"{method.title()} correlation"),
        ("Iron–Turbidity", f"{C.corr_matrix(df, method).loc['Fe', 'TUR']:.2f}", f"{method.title()} correlation"),
        ("Saline-influenced samples", f"{len(saline)}", "Samples with chloride > 250 mg/L (freshwater taste threshold)"),
        ("Max conductivity", f"{df['COND'].max():,.0f}", f"{df.loc[df['COND'].idxmax(), 'location']}"),
    ])

    c1, c2 = st.columns(2)
    c1.plotly_chart(C.fig_corr_heatmap(df, method), width="stretch")
    with c2:
        xa, ya, lg = st.columns([1, 1, 1])
        x = xa.selectbox("X axis", list(C.PARAMS), format_func=C.SHORT.get, index=0)
        y = ya.selectbox("Y axis", list(C.PARAMS), format_func=C.SHORT.get, index=4)
        log = lg.toggle("Log-log + fit", value=True)
        st.plotly_chart(C.fig_scatter_pair(df, x, y, log), width="stretch")
    st.plotly_chart(C.fig_cond_chloride(df), width="stretch")

    interpretation(
        "<b>Sediment indicators move together.</b> Turbidity and suspended solids are strongly correlated "
        "(Spearman ρ = 0.70; log-log slope 0.57, R² = 0.51), and iron is moderately linked to turbidity (ρ = 0.58), "
        "so iron is largely carried on suspended particles. Turbidity rises less than proportionally with SS "
        "(slope below 1) because particle size and colour also affect light scattering. "
        "<br><b>Ionic and nutrient indicators form a second group.</b> Conductivity correlates strongly with chloride "
        "(ρ = 0.78), and both correlate moderately with NH3-N and phosphate (ρ = 0.40–0.54), the signature of "
        "wastewater and saline inputs that concentrate in Skudai. These variables are unrelated or negatively related "
        "to SS and turbidity (ρ between −0.37 and +0.14), so sediment and wastewater are separate pressures "
        "on the rivers. "
        "<br><b>Anomalies:</b> four samples exceed 250 mg/L chloride: Kg. Teluk Serdang (4,150 mg/L; conductivity "
        "13,184), Jeti Tg. Selabu (1,700) and Jeti Pekan Jorak (754 and 408). Their conductivity is also extreme, "
        "and they sit at jetty and downstream locations, which is consistent with tidal seawater intrusion rather "
        "than pollution. They inflate the Skudai and Muar averages for conductivity and chloride, so these "
        "samples should be interpreted separately."
    )

# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------
pages = [
    st.Page(page_spatial, title="1 · Spatial hotspots", icon="🗺️", default=True),
    st.Page(page_temporal, title="2 · Seasonal variation", icon="📈"),
    st.Page(page_relationships, title="3 · Relationships & anomalies", icon="🔗"),
]
with st.sidebar:
    st.markdown("## 💧 Johor River Water Quality 2020")
    st.caption("JIE42403 Scientific Visualisation · 39 stations · 234 samples")
pg = st.navigation(pages)
pg.run()
