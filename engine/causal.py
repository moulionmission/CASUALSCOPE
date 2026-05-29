"""
engine/causal.py
Causal inference pipeline for policy evaluation.
Methods: Difference-in-Differences, Propensity Score Matching,
         Parallel Trends Test, DoWhy causal graph estimation.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.special import expit
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ── 1. Difference-in-Differences ─────────────────────────────────────────────

def run_did(df: pd.DataFrame) -> dict:
    """
    Run two-way fixed effects DiD regression.
    Model: outcome ~ treated*post + state_FE + year_FE + covariates
    """
    covariates = ["median_income_k", "pct_urban", "pct_college",
                  "unemployment_rate", "pct_white"]
    cov_str = " + ".join(covariates)

    # Two-way FE DiD with covariates
    formula = f"outcome ~ treated:post + {cov_str} + C(state) + C(year)"
    model   = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["state"]}
    )

    coef  = model.params.get("treated:post", np.nan)
    se    = model.bse.get("treated:post", np.nan)
    pval  = model.pvalues.get("treated:post", np.nan)
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    # Simple 2x2 DiD for transparency
    pre_treat  = df[(df.treated==1) & (df.post==0)].outcome.mean()
    post_treat = df[(df.treated==1) & (df.post==1)].outcome.mean()
    pre_ctrl   = df[(df.treated==0) & (df.post==0)].outcome.mean()
    post_ctrl  = df[(df.treated==0) & (df.post==1)].outcome.mean()
    naive_did  = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)

    return {
        "method":       "Difference-in-Differences (TWFE)",
        "estimate":     round(coef,  4),
        "std_error":    round(se,    4),
        "p_value":      round(pval,  4),
        "ci_lower":     round(ci_lo, 4),
        "ci_upper":     round(ci_hi, 4),
        "significant":  pval < 0.05,
        "r_squared":    round(model.rsquared, 4),
        "n_obs":        len(df),
        "naive_did":    round(naive_did, 4),
        "pre_treat":    round(pre_treat,  3),
        "post_treat":   round(post_treat, 3),
        "pre_ctrl":     round(pre_ctrl,   3),
        "post_ctrl":    round(post_ctrl,  3),
        "model":        model,
    }


# ── 2. Parallel Trends Test ───────────────────────────────────────────────────

def test_parallel_trends(df: pd.DataFrame) -> dict:
    """
    Test parallel pre-trends assumption.
    Regress outcome on treated*year_dummies in pre-period only.
    If no significant interaction → parallel trends hold.
    """
    pre_df = df[df.post == 0].copy()
    years  = sorted(pre_df.year.unique())

    if len(years) < 2:
        return {"passed": None, "message": "Not enough pre-period years to test."}

    base_year = years[0]
    pre_df["year_num"] = pre_df["year"] - base_year

    # Test: treated × time_trend in pre-period
    formula = "outcome ~ treated * year_num + C(state)"
    model   = smf.ols(formula, data=pre_df).fit(
        cov_type="cluster", cov_kwds={"groups": pre_df["state"]}
    )

    interaction_key = "treated:year_num"
    coef = model.params.get(interaction_key, np.nan)
    pval = model.pvalues.get(interaction_key, np.nan)

    # Year-by-year pre-trend estimates
    yearly = []
    for yr in years[1:]:
        sub = pre_df[pre_df.year.isin([years[0], yr])]
        if len(sub.treated.unique()) < 2:
            continue
        t_mean = sub[sub.treated==1].outcome.mean()
        c_mean = sub[sub.treated==0].outcome.mean()
        yearly.append({"year": yr, "diff": round(t_mean - c_mean, 3)})

    passed = pval > 0.05 if not np.isnan(pval) else None

    return {
        "passed":         passed,
        "interaction_coef": round(coef, 4) if not np.isnan(coef) else None,
        "p_value":        round(pval, 4) if not np.isnan(pval) else None,
        "message": (
            "✅ Parallel trends assumption holds (p={:.3f}) — pre-period trends are not significantly different.".format(pval)
            if passed else
            "⚠️ Parallel trends may be violated (p={:.3f}) — interpret DiD estimates with caution.".format(pval)
            if passed is not None else "Could not test."
        ),
        "yearly_diffs": yearly,
    }


# ── 3. Propensity Score Matching ──────────────────────────────────────────────

def run_psm(df: pd.DataFrame) -> dict:
    """
    Propensity score matching on pre-period state characteristics.
    Matches treated states to control states on observable covariates.
    Returns ATT estimate from matched sample.
    """
    covariates = ["median_income_k", "pct_urban", "pct_college",
                  "pct_white", "unemployment_rate"]

    # Use pre-period baseline (one row per state)
    baseline = (df[df.post == 0]
                .groupby("state")[covariates + ["treated", "outcome"]]
                .mean()
                .reset_index())

    if baseline.treated.sum() < 3 or (1 - baseline.treated).sum() < 3:
        return {"error": "Not enough states in each group for PSM."}

    X = baseline[covariates].values
    T = baseline["treated"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Estimate propensity scores
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1]
    baseline["propensity_score"] = ps

    # 1:1 nearest-neighbor matching without replacement
    treated_idx   = np.where(T == 1)[0]
    control_idx   = np.where(T == 0)[0]
    treated_ps    = ps[treated_idx].reshape(-1, 1)
    control_ps    = ps[control_idx].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_ps)
    distances, indices = nn.kneighbors(treated_ps)

    matched_control_idx = control_idx[indices.flatten()]

    # Check common support (caliper = 0.05)
    caliper = 0.05
    valid   = distances.flatten() < caliper
    n_matched = valid.sum()

    treated_outcomes  = baseline.iloc[treated_idx[valid]]["outcome"].values
    control_outcomes  = baseline.iloc[matched_control_idx[valid]]["outcome"].values

    if n_matched < 3:
        return {"error": f"Too few matches within caliper ({n_matched}). Try a different policy."}

    att       = (treated_outcomes - control_outcomes).mean()
    att_se    = (treated_outcomes - control_outcomes).std() / np.sqrt(n_matched)
    t_stat    = att / att_se
    p_val     = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_matched - 1))

    # SMD before/after matching
    def smd(x1, x2):
        return abs(x1.mean() - x2.mean()) / np.sqrt((x1.std()**2 + x2.std()**2) / 2 + 1e-9)

    smd_before = [smd(X[T==1, i], X[T==0, i]) for i in range(X.shape[1])]
    smd_after  = [smd(X[treated_idx[valid], i], X[matched_control_idx[valid], i]) for i in range(X.shape[1])]

    return {
        "method":          "Propensity Score Matching (1:1 NN, caliper=0.05)",
        "estimate":        round(att, 4),
        "std_error":       round(att_se, 4),
        "p_value":         round(p_val, 4),
        "ci_lower":        round(att - 1.96 * att_se, 4),
        "ci_upper":        round(att + 1.96 * att_se, 4),
        "significant":     p_val < 0.05,
        "n_matched_pairs": int(n_matched),
        "n_treated":       len(treated_idx),
        "balance": {
            "covariates": covariates,
            "smd_before": [round(s, 3) for s in smd_before],
            "smd_after":  [round(s, 3) for s in smd_after],
        },
        "baseline_df": baseline,
    }


# ── 4. DoWhy Causal Graph Estimation ─────────────────────────────────────────

def run_dowhy(df: pd.DataFrame) -> dict:
    """
    Use DoWhy to estimate causal effect with explicit causal graph.
    Identifies effect via backdoor criterion and estimates with linear regression.
    """
    try:
        import dowhy
        from dowhy import CausalModel

        # Use post-period cross-section for DoWhy
        post_df = df[df.post == 1].copy()

        covariates = ["median_income_k", "pct_urban", "pct_college",
                      "pct_white", "unemployment_rate"]

        # Causal graph: treatment ← covariates → outcome; treatment → outcome
        causal_graph = """
        digraph {
            treated -> outcome;
            median_income_k -> treated;
            median_income_k -> outcome;
            pct_urban -> treated;
            pct_urban -> outcome;
            pct_college -> treated;
            pct_college -> outcome;
            pct_white -> treated;
            pct_white -> outcome;
            unemployment_rate -> treated;
            unemployment_rate -> outcome;
        }
        """

        model = CausalModel(
            data=post_df,
            treatment="treated",
            outcome="outcome",
            graph=causal_graph.strip(),
        )

        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate   = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
            control_value=0,
            treatment_value=1,
        )

        # Refutation: placebo treatment test
        refute = model.refute_estimate(
            identified, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=100,
        )

        return {
            "method":           "DoWhy — Backdoor Criterion (Linear Regression)",
            "estimate":         round(float(estimate.value), 4),
            "identified":       True,
            "estimand":         str(identified.estimands.get("backdoor", "backdoor"))[:200],
            "refutation_value": round(float(refute.new_effect), 4),
            "refutation_passed": abs(float(refute.new_effect)) < abs(float(estimate.value)) * 0.3,
            "refutation_note":  "Placebo test: new effect should be near zero if causal estimate is valid.",
        }

    except Exception as e:
        return {
            "method":    "DoWhy — Backdoor Criterion",
            "estimate":  None,
            "error":     str(e)[:200],
            "identified": False,
        }


# ── 5. Sensitivity Analysis ───────────────────────────────────────────────────

def sensitivity_analysis(df: pd.DataFrame, did_result: dict) -> dict:
    """
    Rosenbaum-style sensitivity: how strong would unmeasured confounding
    need to be to explain away the observed effect?
    Uses E-value approximation.
    """
    est  = abs(did_result["estimate"])
    se   = did_result["std_error"]

    if se == 0 or np.isnan(est):
        return {"e_value": None, "interpretation": "Cannot compute."}

    # E-value for mean difference (VanderWeele & Ding approximation)
    # For continuous outcomes, use standardized effect
    outcome_sd = df.outcome.std()
    std_effect = est / outcome_sd if outcome_sd > 0 else 0

    # Convert to approximate RR for E-value
    rr_approx = np.exp(0.91 * std_effect)
    e_value   = rr_approx + np.sqrt(rr_approx * (rr_approx - 1))

    return {
        "e_value": round(e_value, 3),
        "std_effect": round(std_effect, 4),
        "interpretation": (
            f"E-value = {e_value:.2f}. An unmeasured confounder would need to be associated "
            f"with both the treatment and outcome by a risk ratio of {e_value:.2f}-fold "
            f"to fully explain away the observed effect. "
            + ("This is a strong effect — unlikely to be confounded away."
               if e_value > 2.5 else
               "Moderate — some unmeasured confounding could affect conclusions.")
        ),
    }


# ── Master pipeline ───────────────────────────────────────────────────────────

def run_full_pipeline(df: pd.DataFrame, policy: dict) -> dict:
    """Run all methods and return unified results."""
    results = {}
    results["did"]              = run_did(df)
    results["parallel_trends"]  = test_parallel_trends(df)
    results["psm"]              = run_psm(df)
    results["dowhy"]            = run_dowhy(df)
    results["sensitivity"]      = sensitivity_analysis(df, results["did"])
    results["policy"]           = policy
    return results
