"""
engine/plots.py
All visualizations for the causal inference dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BLUE   = "#2A5C8B"
ORANGE = "#E07B39"
GREEN  = "#26A69A"
RED    = "#C0392B"
GRAY   = "#95A5A6"
BG     = "#FAFAF8"
GRID   = "#E8E8E4"

def _base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(size=14, color="#1C1C1A"), x=0.01),
        paper_bgcolor=BG, plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#444"),
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(showgrid=True, gridcolor=GRID, linecolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID, linecolor=GRID),
    )


def plot_trends(df: pd.DataFrame, policy: dict) -> go.Figure:
    """Mean outcome over time for treated vs control groups with policy line."""
    yearly = (df.groupby(["year", "treated"])["outcome"]
                .agg(["mean","sem"])
                .reset_index())
    yearly.columns = ["year","treated","mean","sem"]

    fig = go.Figure()
    for grp, name, color in [(1,"Treated",BLUE),(0,"Control",ORANGE)]:
        sub = yearly[yearly.treated==grp]
        fig.add_trace(go.Scatter(
            x=sub.year, y=sub["mean"],
            error_y=dict(type="data", array=sub["sem"]*1.96, visible=True,
                         color=color, thickness=1.5, width=4),
            mode="lines+markers",
            name=name, line=dict(color=color, width=2.5),
            marker=dict(size=7),
        ))

    py = policy["policy_year"]
    fig.add_vline(x=py - 0.5, line_dash="dash", line_color=RED, line_width=1.5)
    fig.add_annotation(x=py - 0.5, y=yearly["mean"].max(),
                       text=f"Policy: {py}", showarrow=False,
                       font=dict(color=RED, size=10), xshift=45)

    layout = _base_layout(f"Mean {policy['outcome_label']} Over Time")
    layout["legend"] = dict(orientation="h", y=-0.15)
    fig.update_layout(**layout)
    return fig


def plot_parallel_trends(df: pd.DataFrame, policy: dict) -> go.Figure:
    """Pre-period trends only — visual test for parallel trends assumption."""
    pre = df[df.post==0]
    yearly = (pre.groupby(["year","treated"])["outcome"]
                 .mean().reset_index())

    fig = go.Figure()
    for grp, name, color, dash in [(1,"Treated",BLUE,"solid"),(0,"Control",ORANGE,"dot")]:
        sub = yearly[yearly.treated==grp]
        fig.add_trace(go.Scatter(
            x=sub.year, y=sub["outcome"],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=7),
        ))

    layout = _base_layout("Pre-Period Parallel Trends Test")
    layout["xaxis"]["title"] = "Year (pre-policy only)"
    layout["yaxis"]["title"] = policy["outcome_label"]
    layout["annotations"] = [dict(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text="Lines should be roughly parallel if assumption holds",
        showarrow=False, font=dict(size=10, color=GRAY)
    )]
    fig.update_layout(**layout)
    return fig


def plot_estimates(results: dict) -> go.Figure:
    """Forest plot comparing estimates across methods."""
    methods, estimates, lowers, uppers, colors = [], [], [], [], []

    for key, label in [("did","DiD (TWFE)"),("psm","Prop. Score Matching"),("dowhy","DoWhy (Backdoor)")]:
        r = results.get(key, {})
        est = r.get("estimate")
        if est is None: continue
        lo  = r.get("ci_lower", est - 0.5)
        hi  = r.get("ci_upper", est + 0.5)
        sig = r.get("significant", False)
        methods.append(label)
        estimates.append(est)
        lowers.append(abs(est - lo))
        uppers.append(abs(hi - est))
        colors.append(BLUE if sig else GRAY)

    fig = go.Figure()
    for i, (m, e, lo, hi, c) in enumerate(zip(methods, estimates, lowers, uppers, colors)):
        fig.add_trace(go.Scatter(
            x=[e], y=[m],
            error_x=dict(type="data", symmetric=False,
                         array=[hi], arrayminus=[lo],
                         color=c, thickness=2, width=8),
            mode="markers",
            marker=dict(size=12, color=c, symbol="diamond"),
            name=m, showlegend=False,
        ))

    fig.add_vline(x=0, line_dash="dash", line_color=RED, line_width=1)
    layout = _base_layout("Causal Effect Estimates — All Methods")
    layout["xaxis"]["title"] = "Estimated Effect (with 95% CI)"
    layout["yaxis"]["autorange"] = "reversed"
    layout["height"] = 280
    fig.update_layout(**layout)
    return fig


def plot_psm_balance(psm_result: dict) -> go.Figure:
    """Covariate balance before and after PSM (SMD plot)."""
    if "error" in psm_result or "balance" not in psm_result:
        fig = go.Figure()
        fig.add_annotation(text="PSM balance not available",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=12, color=GRAY))
        fig.update_layout(**_base_layout())
        return fig

    bal  = psm_result["balance"]
    covs = [c.replace("_", " ").title() for c in bal["covariates"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=covs, x=bal["smd_before"],
        orientation="h", name="Before Matching",
        marker_color=ORANGE, opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        y=covs, x=bal["smd_after"],
        orientation="h", name="After Matching",
        marker_color=GREEN, opacity=0.9,
    ))
    fig.add_vline(x=0.1, line_dash="dash", line_color=RED, line_width=1)
    fig.add_annotation(x=0.1, y=len(covs)-0.5,
                       text="SMD = 0.1 threshold", showarrow=False,
                       font=dict(size=9, color=RED), xshift=55)

    layout = _base_layout("Covariate Balance — Before vs After PSM")
    layout["barmode"]  = "group"
    layout["xaxis"]["title"] = "Standardized Mean Difference (SMD)"
    layout["legend"]   = dict(orientation="h", y=-0.2)
    layout["height"]   = 320
    fig.update_layout(**layout)
    return fig


def plot_state_map(df: pd.DataFrame, policy: dict) -> go.Figure:
    """Choropleth showing treated vs control states."""
    states = df[["state","treated"]].drop_duplicates()
    states["label"] = states.treated.map({1:"Treated", 0:"Control"})

    fig = go.Figure(go.Choropleth(
        locations=states.state,
        z=states.treated,
        locationmode="USA-states",
        colorscale=[[0, "#F5E6D3"], [1, BLUE]],
        showscale=False,
        text=states.apply(lambda r: f"{r.state}: {r.label}", axis=1),
        hoverinfo="text",
    ))
    layout = _base_layout(f"Treatment Assignment — {policy['name']}")
    layout["geo"] = dict(scope="usa", projection_type="albers usa",
                         showlakes=False, bgcolor=BG,
                         lakecolor=BG, landcolor="#F0EDE8",
                         coastlinecolor=GRID, subunitcolor=GRID)
    layout["height"]   = 320
    layout["margin"]   = dict(l=0, r=0, t=40, b=0)
    fig.update_layout(**layout)
    return fig


def plot_event_study(df: pd.DataFrame, policy: dict) -> go.Figure:
    """
    Event study plot — year-by-year DiD estimates relative to policy year.
    The gold standard visualization in causal inference papers.
    """
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")

    py      = policy["policy_year"]
    years   = sorted(df.year.unique())
    base_yr = py - 1  # omit year before policy as baseline

    df2 = df.copy()
    df2["yr_rel"] = df2.year - py

    coefs, cis, yr_rels = [], [], []

    for yr in years:
        if yr == base_yr: continue
        yr_rel = yr - py
        sub = df2[df2.year.isin([base_yr, yr])].copy()
        sub["indicator"] = (sub.year == yr).astype(int)
        try:
            m = smf.ols(
                "outcome ~ treated:indicator + C(state) + median_income_k + unemployment_rate",
                data=sub
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["state"]})
            k = "treated:indicator"
            if k in m.params:
                coefs.append(m.params[k])
                cis.append(1.96 * m.bse[k])
                yr_rels.append(yr_rel)
        except Exception:
            pass

    if not coefs:
        fig = go.Figure()
        fig.add_annotation(text="Event study not available",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=12, color=GRAY))
        fig.update_layout(**_base_layout())
        return fig

    colors = [RED if yr >= 0 else GRAY for yr in yr_rels]

    fig = go.Figure()
    for i, (yr, c, ci, col) in enumerate(zip(yr_rels, coefs, cis, colors)):
        fig.add_trace(go.Scatter(
            x=[yr], y=[c],
            error_y=dict(type="data", array=[ci], color=col,
                         thickness=1.5, width=5),
            mode="markers",
            marker=dict(size=9, color=col),
            showlegend=False,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=GRAY, line_width=1)
    fig.add_vline(x=-0.5, line_dash="dash", line_color=RED, line_width=1.5,
                  annotation_text="Policy enacted", annotation_font_color=RED)

    layout = _base_layout("Event Study — Year-by-Year Treatment Effects")
    layout["xaxis"]["title"] = "Years Relative to Policy"
    layout["yaxis"]["title"] = "Estimated Effect"
    layout["xaxis"]["tickvals"] = yr_rels
    fig.update_layout(**layout)
    return fig
