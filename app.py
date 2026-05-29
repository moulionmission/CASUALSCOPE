import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import load_policy_data, POLICIES
from engine.causal import run_full_pipeline
from engine.plots  import (plot_trends, plot_parallel_trends, plot_estimates,
                            plot_psm_balance, plot_state_map, plot_event_study)

st.set_page_config(page_title="CausalScope", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.stApp{background:#FAFAF8!important;}
.block-container{padding:1.5rem 2rem 3rem!important; max-width:1200px!important;}
#MainMenu,footer{visibility:hidden!important;}

.wordmark{font-family:'Instrument Serif',serif!important;font-size:1.5rem;color:#1C1C1A;letter-spacing:-0.3px;}
.tagline{font-size:0.72rem;color:#999;text-transform:uppercase;letter-spacing:0.06em;font-weight:400;margin-bottom:1.5rem;display:block;}

.result-card{background:white;border:1px solid #E4E4E0;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:0.7rem;}
.result-card.sig{border-left:3px solid #2A5C8B;}
.result-card.insig{border-left:3px solid #95A5A6;}
.result-card.warn{border-left:3px solid #E07B39;}

.metric-row{display:flex;gap:1.5rem;flex-wrap:wrap;margin:0.8rem 0;}
.metric-box{background:#F5F5F3;border-radius:8px;padding:0.7rem 1rem;min-width:110px;text-align:center;}
.metric-val{font-size:1.3rem;font-weight:600;color:#1C1C1A;font-family:'Instrument Serif',serif!important;}
.metric-lbl{font-size:0.68rem;color:#888;text-transform:uppercase;letter-spacing:0.04em;margin-top:2px;}

.verdict-yes{background:#E8F5E9;border:1px solid #A5D6A7;border-radius:8px;padding:0.8rem 1.1rem;font-size:0.88rem;color:#1B5E20;margin:0.8rem 0;}
.verdict-no{background:#FFF3E0;border:1px solid #FFCC80;border-radius:8px;padding:0.8rem 1.1rem;font-size:0.88rem;color:#E65100;margin:0.8rem 0;}
.verdict-mixed{background:#E3F2FD;border:1px solid #90CAF9;border-radius:8px;padding:0.8rem 1.1rem;font-size:0.88rem;color:#0D47A1;margin:0.8rem 0;}

.method-badge{display:inline-block;background:#EFF5FB;color:#2A5C8B;border:1px solid #B3D0EE;border-radius:5px;padding:1px 9px;font-size:0.71rem;font-weight:500;margin-right:4px;}
.section-title{font-size:0.95rem;font-weight:600;color:#1C1C1A;margin:1.2rem 0 0.5rem;}
.explainer{font-size:0.78rem;color:#777;line-height:1.6;margin-bottom:0.8rem;}

section[data-testid="stSidebar"]{background:#F0F0EC!important;border-right:1px solid #E4E4E0!important;}
div[data-testid="stButton"]>button{background:#1C1C1A!important;color:white!important;border:none!important;border-radius:8px!important;font-size:0.82rem!important;font-weight:500!important;padding:0.5rem 1.2rem!important;}
div[data-testid="stButton"]>button:hover{background:#2A5C8B!important;}
div[data-testid="stSelectbox"]>div>div{border-radius:8px!important;font-size:0.84rem!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state: st.session_state.results = None
if "df"      not in st.session_state: st.session_state.df      = None
if "policy"  not in st.session_state: st.session_state.policy  = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="wordmark">🔬 CausalScope</span>', unsafe_allow_html=True)
    st.markdown('<span class="tagline">Policy evaluation · causal inference · observational health data</span>', unsafe_allow_html=True)

    st.markdown("**Select a policy intervention:**")
    policy_labels = {k: v["name"] for k, v in POLICIES.items()}
    selected_key  = st.selectbox("Policy", list(policy_labels.keys()),
                                  format_func=lambda k: policy_labels[k],
                                  label_visibility="collapsed")
    selected_policy = POLICIES[selected_key]

    st.markdown(f"""
    <div style="background:white;border:1px solid #E4E4E0;border-radius:10px;padding:0.9rem 1rem;margin:0.5rem 0;font-size:0.78rem;color:#555;line-height:1.6;">
    <b style="color:#1C1C1A;">{selected_policy['name']}</b><br>
    {selected_policy['description']}<br><br>
    <b>Outcome:</b> {selected_policy['outcome_label']}<br>
    <b>Policy year:</b> {selected_policy['policy_year']}<br>
    <b>Treated states:</b> {len(selected_policy['treated_states'])}<br>
    <b>Years:</b> {selected_policy['years'][0]}–{selected_policy['years'][-1]}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶ Run Causal Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#aaa;line-height:1.7;">
    <b style="color:#666;">Methods used:</b><br>
    · Difference-in-Differences (TWFE)<br>
    · Parallel Trends Testing<br>
    · Propensity Score Matching<br>
    · DoWhy Causal Graph<br>
    · Event Study Design<br>
    · E-value Sensitivity Analysis
    </div>""", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<span class="wordmark">CausalScope</span>', unsafe_allow_html=True)
st.markdown('<span class="tagline">Estimating causal effects of public health policy on population outcomes</span>', unsafe_allow_html=True)

if run_btn:
    with st.spinner("Running causal inference pipeline..."):
        df, policy = load_policy_data(selected_key)
        results    = run_full_pipeline(df, policy)
        st.session_state.df      = df
        st.session_state.policy  = policy
        st.session_state.results = results
    st.success("Analysis complete.")

if st.session_state.results is None:
    st.info("Select a policy from the sidebar and click **Run Causal Analysis** to begin.")
    st.markdown("""
    <div style="background:white;border:1px solid #E4E4E0;border-radius:12px;padding:1.4rem 1.6rem;margin-top:1rem;">
    <div class="section-title">How this works</div>
    <div class="explainer">
    This tool estimates the <b>causal effect</b> of a real US public health policy using a multi-method pipeline.<br><br>
    Unlike simple before/after comparisons, it uses <b>causal inference methods</b> to separate the policy's true effect
    from confounding trends — the same methodological challenge at the heart of observational epidemiology research.<br><br>
    <b>Methods:</b> Difference-in-Differences isolates the treatment effect using a control group.
    Parallel trends testing validates the core assumption. Propensity score matching balances
    observable confounders. DoWhy encodes the causal graph explicitly. E-values quantify
    sensitivity to unmeasured confounding.
    </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
results = st.session_state.results
df      = st.session_state.df
policy  = st.session_state.policy
did     = results["did"]
psm     = results["psm"]
pt      = results["parallel_trends"]
dw      = results["dowhy"]
sens    = results["sensitivity"]

# ── VERDICT ───────────────────────────────────────────────────────────────────
n_sig = sum([
    did.get("significant", False),
    psm.get("significant", False),
    dw.get("estimate") is not None and abs(dw.get("estimate",0)) > 0.3,
])
all_same_direction = all(
    r.get("estimate", 0) * did.get("estimate", 1) > 0
    for r in [psm, dw] if r.get("estimate") is not None
)

if n_sig >= 2 and all_same_direction:
    verdict_class = "verdict-yes"
    verdict_icon  = "✅"
    verdict_text  = f"<b>Evidence supports a causal effect.</b> {n_sig}/3 methods detect a statistically significant effect in the same direction. The policy is associated with a meaningful change in {policy['outcome_label'].lower()}."
elif n_sig == 1:
    verdict_class = "verdict-mixed"
    verdict_icon  = "⚠️"
    verdict_text  = f"<b>Mixed evidence.</b> Only 1 of 3 methods reaches significance. The effect may be real but is sensitive to methodological choices. Interpret with caution."
else:
    verdict_class = "verdict-no"
    verdict_icon  = "❌"
    verdict_text  = f"<b>Insufficient evidence.</b> No consistent causal effect detected across methods. The policy may not have had a statistically detectable impact on {policy['outcome_label'].lower()}."

st.markdown(f'<div class="{verdict_class}">{verdict_icon} {verdict_text}</div>', unsafe_allow_html=True)

# ── KEY METRICS ───────────────────────────────────────────────────────────────
est   = did["estimate"]
ci_lo = did["ci_lower"]
ci_hi = did["ci_upper"]
pval  = did["p_value"]
eval  = sens.get("e_value","—")

st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="metric-val">{est:+.3f}</div><div class="metric-lbl">DiD Estimate</div></div>
  <div class="metric-box"><div class="metric-val">[{ci_lo:.2f}, {ci_hi:.2f}]</div><div class="metric-lbl">95% CI</div></div>
  <div class="metric-box"><div class="metric-val">p={pval:.3f}</div><div class="metric-lbl">p-value (clustered)</div></div>
  <div class="metric-box"><div class="metric-val">{eval}</div><div class="metric-lbl">E-value</div></div>
  <div class="metric-box"><div class="metric-val">{did['n_obs']}</div><div class="metric-lbl">Observations</div></div>
</div>""", unsafe_allow_html=True)

# ── PARALLEL TRENDS ───────────────────────────────────────────────────────────
pt_color = "#E8F5E9" if pt.get("passed") else "#FFF3E0"
pt_border = "#A5D6A7" if pt.get("passed") else "#FFCC80"
pt_text_color = "#1B5E20" if pt.get("passed") else "#E65100"
st.markdown(f"""
<div style="background:{pt_color};border:1px solid {pt_border};border-radius:8px;
     padding:0.6rem 1rem;font-size:0.8rem;color:{pt_text_color};margin:0.5rem 0;">
<b>Parallel Trends:</b> {pt.get("message","Not tested")}
</div>""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📈 Trends & Map",
    "🔬 Event Study",
    "⚖️ All Methods",
    "🎯 PSM Balance",
    "📋 Interpretation"
])

with t1:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(plot_trends(df, policy), use_container_width=True)
        st.plotly_chart(plot_parallel_trends(df, policy), use_container_width=True)
    with c2:
        st.plotly_chart(plot_state_map(df, policy), use_container_width=True)
        # DiD 2x2 table
        st.markdown('<div class="section-title">DiD 2×2 Decomposition</div>', unsafe_allow_html=True)
        did_table = pd.DataFrame({
            "":        ["Pre-policy", "Post-policy", "Δ (change)"],
            "Treated": [f"{did['pre_treat']:.2f}", f"{did['post_treat']:.2f}",
                        f"{did['post_treat']-did['pre_treat']:.2f}"],
            "Control": [f"{did['pre_ctrl']:.2f}",  f"{did['post_ctrl']:.2f}",
                        f"{did['post_ctrl']-did['pre_ctrl']:.2f}"],
            "DiD":     ["","", f"{did['naive_did']:.2f}"],
        })
        st.dataframe(did_table, hide_index=True, use_container_width=True)

with t2:
    st.markdown('<div class="explainer">The event study is the gold standard visualization in modern causal inference. Each point shows the year-by-year treatment effect relative to the year before the policy. Pre-policy estimates should cluster near zero (validating parallel trends). Post-policy estimates show the dynamic treatment effect.</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_event_study(df, policy), use_container_width=True)

with t3:
    st.plotly_chart(plot_estimates(results), use_container_width=True)

    cols = st.columns(3)
    for i, (key, label, r) in enumerate([
        ("did","Difference-in-Differences", did),
        ("psm","Propensity Score Matching", psm),
        ("dowhy","DoWhy (Causal Graph)", dw),
    ]):
        with cols[i]:
            est_val = r.get("estimate")
            sig     = r.get("significant", False)
            card_cls = "result-card sig" if sig else "result-card insig"
            if est_val is None:
                st.markdown(f'<div class="{card_cls}"><b>{label}</b><br><span style="color:#aaa;font-size:0.8rem;">{r.get("error","Not available")}</span></div>', unsafe_allow_html=True)
            else:
                lo = r.get("ci_lower", "—")
                hi = r.get("ci_upper", "—")
                p  = r.get("p_value",  "—")
                st.markdown(f"""
                <div class="{card_cls}">
                  <div style="font-size:0.82rem;font-weight:600;color:#1C1C1A;margin-bottom:0.5rem;">{label}</div>
                  <div style="font-size:1.3rem;font-weight:600;color:{'#2A5C8B' if sig else '#95A5A6'};">{est_val:+.3f}</div>
                  <div style="font-size:0.72rem;color:#888;margin-top:3px;">95% CI [{lo}, {hi}]</div>
                  <div style="font-size:0.72rem;color:#888;">p = {p} {'✓ sig.' if sig else ''}</div>
                </div>""", unsafe_allow_html=True)

with t4:
    st.markdown('<div class="explainer">Standardized Mean Difference (SMD) measures covariate imbalance between treated and control states. SMD < 0.1 (dashed line) is the conventional threshold for "good balance." After matching, all covariates should fall below this line.</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_psm_balance(psm), use_container_width=True)
    if "n_matched_pairs" in psm:
        st.markdown(f"""
        <div style="font-size:0.8rem;color:#555;background:white;border:1px solid #E4E4E0;border-radius:8px;padding:0.7rem 1rem;">
        <b>Matched pairs:</b> {psm['n_matched_pairs']} of {psm['n_treated']} treated states &nbsp;·&nbsp;
        <b>ATT estimate:</b> {psm['estimate']:+.3f} &nbsp;·&nbsp;
        <b>p-value:</b> {psm['p_value']:.3f}
        </div>""", unsafe_allow_html=True)

with t5:
    st.markdown('<div class="section-title">How to Interpret These Results</div>', unsafe_allow_html=True)

    # DiD interpretation
    dir_word = "decrease" if est < 0 else "increase"
    outcome  = policy["outcome_label"]
    treated_n = len(policy["treated_states"])
    st.markdown(f"""
    <div class="result-card {'sig' if did['significant'] else 'insig'}">
    <b>Main Finding (DiD):</b> The policy was associated with a <b>{abs(est):.2f} unit {dir_word}</b>
    in {outcome} in treated states relative to controls (95% CI: [{ci_lo:.2f}, {ci_hi:.2f}], p={pval:.3f}).
    This estimate controls for state and year fixed effects plus observed covariates, with standard errors
    clustered at the state level (n={treated_n} treated states).
    {"<br><b>This effect is statistically significant at the 5% level.</b>" if did['significant'] else "<br><i>This effect does not reach statistical significance.</i>"}
    </div>""", unsafe_allow_html=True)

    # Sensitivity
    st.markdown(f"""
    <div class="result-card warn">
    <b>Sensitivity to Unmeasured Confounding (E-value):</b><br>
    {sens.get('interpretation', 'Not available')}
    </div>""", unsafe_allow_html=True)

    # DoWhy refutation
    if dw.get("estimate") is not None and "refutation_value" in dw:
        ref_passed = dw.get("refutation_passed", False)
        st.markdown(f"""
        <div class="result-card {'sig' if ref_passed else 'warn'}">
        <b>DoWhy Refutation Test (Placebo Treatment):</b><br>
        Randomizing treatment assignment produced an effect of {dw['refutation_value']:+.3f}
        (original: {dw['estimate']:+.3f}).
        {"✅ Placebo effect is near zero — supports validity of causal estimate." if ref_passed else "⚠️ Placebo effect is non-trivial — interpret with caution."}
        </div>""", unsafe_allow_html=True)

    # Limitations
    st.markdown("""
    <div style="background:#F5F5F3;border-radius:8px;padding:0.9rem 1.1rem;font-size:0.78rem;color:#666;line-height:1.7;margin-top:0.8rem;">
    <b style="color:#1C1C1A;">Limitations:</b><br>
    · State-level analysis may mask within-state heterogeneity<br>
    · Only observed confounders are controlled — unmeasured confounding may remain<br>
    · SUTVA assumes no spillover effects between states<br>
    · Staggered adoption (states treated at different times) requires more complex estimators (Callaway-Sant'Anna)<br>
    · Data represents population averages — individual-level effects may differ
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;font-size:0.68rem;color:#ccc;padding:2rem 0 0.5rem;">
Policy Causal Engine · Built with DoWhy, EconML, statsmodels, lifelines · Open source
</div>""", unsafe_allow_html=True)