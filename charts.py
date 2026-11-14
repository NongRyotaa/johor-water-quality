"""Chart builders shared by the Streamlit app (app.py) and the PDF report generator."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_FILE = Path(__file__).parent / "johor_water_quality_2020_clean.csv"

PARAMS = {
    "SS": "Suspended solids (mg/L)",
    "NH3N": "Ammoniacal nitrogen, NH3-N (mg/L)",
    "TEMP": "Water temperature (°C)",
    "COND": "Conductivity (µS/cm*)",
    "TUR": "Turbidity (NTU)",
    "Cl": "Chloride (mg/L)",
    "PO4": "Phosphate (mg/L)",
    "Fe": "Iron (mg/L)",
}
SHORT = {"SS": "SS", "NH3N": "NH3-N", "TEMP": "Temp", "COND": "Cond", "TUR": "Turbidity",
         "Cl": "Chloride", "PO4": "PO4", "Fe": "Iron", "As": "As"}
BASIN_COLORS = {"Muar": "#1f77b4", "Skudai": "#d62728", "Johor": "#2ca02c", "Linggi": "#ff7f0e"}
BASIN_ORDER = ["Muar", "Skudai", "Johor", "Linggi"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# NWQS Class III limits (used as reference lines)
NWQS_III = {"NH3N": 0.9, "SS": 150}

TEMPLATE = "plotly_white"


def load_data(path=DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["basin"] = pd.Categorical(df["basin"], categories=BASIN_ORDER, ordered=True)
    return df


def station_table(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """One row per station with mean / max of the chosen parameter."""
    g = (df.groupby(["station_id", "location", "basin", "lat", "lon"], observed=True)[param]
           .agg(mean="mean", max="max", n="count").reset_index())
    short = g["location"].str.title().apply(lambda t: t if len(t) <= 24 else t[:23].rstrip(" ,.") + "…")
    g["label"] = g["station_id"] + " · " + short
    return g


def _log(fig, x=False, y=False):
    """Log axis showing only powers of ten, with faint minor grid."""
    kw = dict(type="log", dtick=1, minor=dict(showgrid=False))
    if x:
        fig.update_xaxes(**kw)
    if y:
        fig.update_yaxes(**kw)
    return fig


def _base(fig, title, height=380):
    fig.update_layout(template=TEMPLATE, title=dict(text=title, x=0.01, font=dict(size=14)),
                      height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(title="", orientation="h", y=-0.2))
    return fig


# --------------------------------------------------------------------------------------
# Objective 1 – spatial patterns
# --------------------------------------------------------------------------------------
def fig_station_map(df, param):
    g = station_table(df, param)
    hi = float(g["mean"].quantile(0.95))
    fig = px.scatter_map(
        g, lat="lat", lon="lon", color="mean", size=np.clip(g["mean"], 0, None) + 1e-9,
        size_max=22, color_continuous_scale="YlOrRd", range_color=(float(g["mean"].min()), hi),
        hover_name="label", hover_data={"basin": True, "mean": ":.2f", "max": ":.2f",
                                         "lat": False, "lon": False},
        zoom=7.3, center=dict(lat=2.35, lon=102.9), map_style="carto-positron",
        labels={"mean": "Mean"})
    fig.update_layout(coloraxis_colorbar=dict(title=f"Mean {SHORT[param]}", thickness=12))
    return _base(fig, f"Station map – mean {PARAMS[param]}", 430)


def fig_basin_box(df, param, log=False):
    fig = px.box(df, x="basin", y=param, color="basin", points="all", color_discrete_map=BASIN_COLORS,
                 category_orders={"basin": BASIN_ORDER}, labels={"basin": "Basin", param: PARAMS[param]},
                 hover_data={"location": True, "date": "|%d %b"})
    fig.update_traces(marker=dict(size=4, opacity=0.6), jitter=0.4, boxmean=True)
    if log:
        _log(fig, y=True)
    fig.update_layout(showlegend=False)
    return _base(fig, f"{SHORT[param]} distribution by basin", 430)


def fig_station_rank(df, param, top=10):
    g = station_table(df, param).nlargest(top, "mean").sort_values("mean")
    fig = px.bar(g, x="mean", y="label", color="basin", orientation="h", color_discrete_map=BASIN_COLORS,
                 category_orders={"basin": BASIN_ORDER}, labels={"mean": f"Mean {PARAMS[param]}", "label": ""},
                 text=g["mean"].round(2))
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="array", categoryarray=g["label"].tolist())
    if param in NWQS_III:
        fig.add_vline(x=NWQS_III[param], line_dash="dash", line_color="black",
                      annotation_text="NWQS Class III", annotation_position="top")
    fig.update_xaxes(range=[0, g["mean"].max() * 1.18])
    return _base(fig, f"Top {top} stations by mean {SHORT[param]}", 380)


# --------------------------------------------------------------------------------------
# Objective 2 – temporal patterns
# --------------------------------------------------------------------------------------
def fig_monthly_lines(df, param):
    """Monthly mean of one parameter per basin + all-station mean. Marker size ~ number of samples."""
    fig = go.Figure()
    groups = [(b, df[df["basin"] == b]) for b in BASIN_ORDER] + [("All stations", df)]
    for name, d in groups:
        if d.empty:
            continue
        m = d.groupby("month")[param].agg(["mean", "count"]).reindex(range(1, 13))
        allst = name == "All stations"
        fig.add_trace(go.Scatter(
            x=MONTHS, y=m["mean"], mode="lines+markers", name=name,
            connectgaps=name in ("Skudai", "Johor", "Linggi"),  # these basins are sampled in alternate months
            customdata=m["count"], line=dict(width=3 if allst else 2, dash="dash" if allst else ("dot" if name in ("Johor", "Linggi") else "solid"),
                                             color="black" if allst else BASIN_COLORS[name]),
            marker=dict(size=np.clip(np.sqrt(m["count"].fillna(0)) * 2.4, 4, 16),
                        color="black" if allst else BASIN_COLORS[name]),
            opacity=1 if name in ("All stations", "Muar", "Skudai") else 0.55,
            hovertemplate=f"{name} – %{{x}}<br>Mean: %{{y:.2f}}<br>Samples: %{{customdata}}<extra></extra>"))
    fig.update_yaxes(title=PARAMS[param])
    fig.update_xaxes(categoryorder="array", categoryarray=MONTHS)
    return _base(fig, f"Monthly mean {SHORT[param]} by basin (marker size = no. of samples)", 380)


ROUND_LABEL = {1: "R1 Jan–Feb", 2: "R2 Mar", 3: "R3 May–Jun", 4: "R4 Jul–Aug", 5: "R5 Sep–Oct", 6: "R6 Nov–Dec"}


def fig_round_heatmap(df, param):
    pv = (df.groupby(["basin", "round"], observed=True)[param].mean().unstack("round")
            .reindex(BASIN_ORDER))
    pv = pv[[c for c in sorted(pv.columns)]]
    z = pv.values
    text = np.where(np.isnan(z), "", np.char.mod("%.2f", np.nan_to_num(z)))
    fig = go.Figure(go.Heatmap(z=z, x=[ROUND_LABEL[c] for c in pv.columns], y=pv.index.astype(str),
                               colorscale="YlGnBu", text=text, texttemplate="%{text}",
                               colorbar=dict(title=SHORT[param], thickness=12),
                               hovertemplate="%{y} · %{x}<br>Mean: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, f"Mean {SHORT[param]} by basin and sampling round", 380)


def fig_timeline(df, param, log=False):
    fig = px.scatter(df, x="date", y=param, color="basin", color_discrete_map=BASIN_COLORS,
                     category_orders={"basin": BASIN_ORDER}, hover_name="location",
                     labels={"date": "Sampling date", param: PARAMS[param]})
    fig.update_traces(marker=dict(size=7, opacity=0.65, line=dict(width=0)))
    med = df.groupby(df["date"].dt.to_period("M").dt.to_timestamp())[param].median().reset_index()
    fig.add_trace(go.Scatter(x=med["date"] + pd.Timedelta(days=14), y=med[param], mode="lines",
                             name="Monthly median", line=dict(color="black", width=2, dash="dot")))
    if param in NWQS_III:
        fig.add_hline(y=NWQS_III[param], line_dash="dash", line_color="grey",
                      annotation_text="NWQS Class III", annotation_position="top left")
    if log:
        _log(fig, y=True)
    return _base(fig, f"Individual {SHORT[param]} readings through 2020", 380)


# --------------------------------------------------------------------------------------
# Objective 3 – relationships and anomalies
# --------------------------------------------------------------------------------------
CORR_VARS = ["SS", "TUR", "Fe", "NH3N", "PO4", "COND", "Cl", "TEMP"]


def corr_matrix(df, method="spearman"):
    return df[CORR_VARS].corr(method=method)


def fig_corr_heatmap(df, method="spearman"):
    c = corr_matrix(df, method)
    labels = [SHORT[v] for v in CORR_VARS]
    fig = go.Figure(go.Heatmap(z=c.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=np.round(c.values, 2), texttemplate="%{text}",
                               colorbar=dict(title=f"{method.title()} ρ" if method == "spearman" else "Pearson r",
                                             thickness=12)))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, f"{method.title()} correlation matrix", 430)


def log_fit(x, y):
    """Least-squares line in log10-log10 space; returns slope, intercept, r2."""
    ok = (x > 0) & (y > 0) & x.notna() & y.notna()
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    slope, icpt = np.polyfit(lx, ly, 1)
    r2 = np.corrcoef(lx, ly)[0, 1] ** 2
    return slope, icpt, r2, lx.min(), lx.max()


def fig_scatter_pair(df, x="SS", y="TUR", log=True):
    d = df.dropna(subset=[x, y])
    fig = px.scatter(d, x=x, y=y, color="basin", color_discrete_map=BASIN_COLORS,
                     category_orders={"basin": BASIN_ORDER}, hover_name="location",
                     labels={x: PARAMS[x], y: PARAMS[y]})
    fig.update_traces(marker=dict(size=7, opacity=0.65))
    if log:
        _log(fig, x=True, y=True)
        s, i, r2, lo, hi = log_fit(d[x], d[y])
        xs = np.linspace(lo, hi, 50)
        fig.add_trace(go.Scatter(x=10 ** xs, y=10 ** (s * xs + i), mode="lines", name="Log-log fit",
                                 line=dict(color="black", dash="dash")))
        fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, align="left",
                           text=f"log-log slope = {s:.2f}<br>R² = {r2:.2f}", bgcolor="rgba(255,255,255,0.8)")
    return _base(fig, f"{SHORT[y]} vs {SHORT[x]}", 380)


def _short_name(loc):
    if "serdang" in loc.lower():
        return "Kg. Teluk Serdang"
    t = loc.title()
    return t if len(t) <= 22 else t[:21].rstrip(" ,.") + "…"


def fig_cond_chloride(df, saline_cl=250):
    d = df.copy()
    lab_hi, lab_lo = f"Chloride > {saline_cl} mg/L", "Freshwater range"
    d["Salinity influence"] = np.where(d["Cl"] > saline_cl, lab_hi, lab_lo)
    fig = px.scatter(d, x="Cl", y="COND", color="Salinity influence",
                     color_discrete_map={lab_hi: "#d62728", lab_lo: "#7f9fbf"},
                     category_orders={"Salinity influence": [lab_lo, lab_hi]},
                     hover_name="location", hover_data={"basin": True, "date": "|%d %b"},
                     labels={"Cl": PARAMS["Cl"], "COND": PARAMS["COND"]})
    fig.update_traces(marker=dict(size=8, opacity=0.75))
    _log(fig, x=True, y=True)
    fig.add_vline(x=saline_cl, line_dash="dash", line_color="grey")
    hi = d[d["Cl"] > saline_cl].sort_values("Cl").tail(3)
    for _, r in hi.iterrows():
        fig.add_annotation(x=np.log10(r["Cl"]), y=np.log10(r["COND"]), text=_short_name(r["location"]),
                           showarrow=True, arrowhead=2, ax=-55, ay=-14, font=dict(size=10))
    return _base(fig, "Conductivity vs chloride – saline-intrusion anomalies", 380)
