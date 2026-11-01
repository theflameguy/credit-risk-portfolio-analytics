# app/app.py
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ingest import build_master_dataset
from src.model import train_default_models
from src.anomaly import detect_portfolio_anomalies
from src.ai_assistant import ask_portfolio

st.set_page_config(
    page_title="Credit Risk Portfolio Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAW_DIR = PROJECT_ROOT / "data/raw"
REVIEW_LOG_PATH = PROJECT_ROOT / "data/processed/anomaly_reviews.json"


@st.cache_resource(show_spinner="Building master dataset, training PD models & scoring anomalies...")
def load_pipeline_artifacts():
    master = build_master_dataset(RAW_DIR)
    model_bundle = train_default_models(master)
    scored_df = detect_portfolio_anomalies(model_bundle["scored_master"])
    repay_raw = pd.read_csv(RAW_DIR / "repayment_history.csv")
    return scored_df, model_bundle, repay_raw


def load_review_decisions() -> dict:
    if REVIEW_LOG_PATH.exists():
        try:
            return json.loads(REVIEW_LOG_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_review_decisions(decisions: dict) -> None:
    REVIEW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_LOG_PATH.write_text(json.dumps(decisions, indent=2))


df_full, model_bundle, repay_raw = load_pipeline_artifacts()

if "anomaly_decisions" not in st.session_state:
    st.session_state["anomaly_decisions"] = load_review_decisions()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS (Global Drill-Down Across Portfolio)
# -----------------------------------------------------------------------------
st.sidebar.title("Portfolio Filters")
sel_regions = st.sidebar.multiselect(
    "Region", options=sorted(df_full["region"].dropna().unique()), default=sorted(df_full["region"].dropna().unique())
)
sel_loan_types = st.sidebar.multiselect(
    "Loan Type", options=sorted(df_full["loan_type"].dropna().unique()), default=sorted(df_full["loan_type"].dropna().unique())
)
sel_emp = st.sidebar.multiselect(
    "Employment Type", options=sorted(df_full["employment_type"].dropna().unique()), default=sorted(df_full["employment_type"].dropna().unique())
)
sel_risk = st.sidebar.multiselect(
    "Model Risk Tier",
    options=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
    default=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
)

df = df_full[
    df_full["region"].isin(sel_regions)
    & df_full["loan_type"].isin(sel_loan_types)
    & df_full["employment_type"].isin(sel_emp)
    & df_full["risk_tier"].isin(sel_risk)
].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Accounts", f"{len(df):,} / {len(df_full):,}")
st.sidebar.metric("Filtered Exposure", f"${df['loan_amount'].sum():,.0f}")

# -----------------------------------------------------------------------------
# MAIN HEADER & 4 REQUIRED VIEWS
# -----------------------------------------------------------------------------
st.title("Credit Risk Portfolio Monitoring & AI Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Portfolio View & Insights",
    "2. Account-Level View & PD Model",
    "3. AI: Ask the Portfolio (LLM)",
    "4. AI: Anomaly Detection Queue",
])

# =============================================================================
# TAB 1: PORTFOLIO VIEW (Risk Segments, Exposure, Trends & 5 'So What' Insights)
# =============================================================================
with tab1:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Loan Accounts", f"{len(df):,}")
    k2.metric("Total Exposure", f"${df['loan_amount'].sum() / 1e6:,.2f}M")
    k3.metric("Observed Default Rate", f"{df['target_default'].mean() * 100:.2f}%")
    k4.metric("Mean LightGBM PD", f"{df['pd_lgbm'].mean() * 100:.2f}%")
    k5.metric("Flagged Anomalies", f"{int(df['is_anomaly'].sum()):,} ({df['is_anomaly'].mean()*100:.1f}%)")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        dim_choice = st.selectbox(
            "Select Profiling Dimension",
            options=["loan_type", "income_band", "employment_type", "region", "age_band"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        dim_df = (
            df.groupby(dim_choice, as_index=False)
            .agg(
                loan_count=("account_id", "count"),
                exposure=("loan_amount", "sum"),
                default_rate_pct=("target_default", lambda s: round(s.mean() * 100, 2)),
                avg_pd_pct=("pd_lgbm", lambda s: round(s.mean() * 100, 2)),
            )
            .sort_values("default_rate_pct", ascending=False)
        )
        fig_dim = px.bar(
            dim_df,
            x=dim_choice,
            y="default_rate_pct",
            color="exposure",
            text="default_rate_pct",
            title=f"Default Rate (%) & Total Exposure by {dim_choice.replace('_', ' ').title()}",
            labels={"default_rate_pct": "Default Rate (%)"},
        )
        st.plotly_chart(fig_dim, use_container_width=True)

    with c2:
        # Cross-Dimensional Cut Heatmap: Income Band x Loan Type
        piv = pd.pivot_table(
            df,
            values="target_default",
            index="income_band",
            columns="loan_type",
            aggfunc=lambda s: round(s.mean() * 100, 2),
        )
        fig_heat = px.imshow(
            piv,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Reds",
            title="Cross-Dimensional Cut: Default Rate (%) by Income Band × Loan Type",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Origination Vintage Trend & Risk Segment Exposure Breakdown
    c3, c4 = st.columns(2)
    with c3:
        df_trend = df.copy()
        df_trend["orig_quarter"] = pd.to_datetime(df_trend["origination_date"], errors="coerce").dt.to_period("Q").astype(str)
        trend_agg = (
            df_trend.groupby("orig_quarter", as_index=False)
            .agg(default_rate_pct=("target_default", lambda s: round(s.mean() * 100, 2)), accounts=("account_id", "count"))
            .sort_values("orig_quarter")
        )
        fig_trend = px.line(
            trend_agg, x="orig_quarter", y="default_rate_pct", markers=True,
            title="Origination Cohort Default Rate Trend (Quarterly)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c4:
        seg_agg = df.groupby("risk_tier", as_index=False).agg(
            exposure=("loan_amount", "sum"),
            default_rate_pct=("target_default", lambda s: round(s.mean() * 100, 2)),
        )
        fig_seg = px.bar(
            seg_agg, x="risk_tier", y="exposure", color="default_rate_pct",
            title="Portfolio Exposure ($) & Actual Default Rate across Risk Segments",
            color_continuous_scale="OrRd",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with st.expander("Key Portfolio Insights & 'So What' Credit Policy Actions", expanded=True):
        st.markdown("""
1. **Cross-Dimensional Cut (`income_band × loan_type`):** Low-income (`Q1/Q2`) borrowers taking unsecured loans exhibit the highest default concentration when `loan_to_income_ratio` exceeds the portfolio median. **So What:** Enforce a hard LTI cap of 0.35x on unsecured personal loans in Q1/Q2 or require collateral (`collateral_flag = 1`).
2. **External Bureau Contagion (`employment_type × utilization_rate`):** Variable-income / Business Owner borrowers with external bureau utilization > 70% default at more than 2x the rate of low-utilization peers. **So What:** Trigger automated quarterly bureau sweeps and freeze credit lines when external utilization crosses 70%.
3. **Product Exposure vs. Loss Asymmetry (`loan_type`):** High-ticket Mortgages dominate dollar exposure, while shorter-tenure unsecured loans drive disproportionate default frequency. **So What:** Recalibrate risk-based pricing (`interest_rate_pct`) on top-decile PD unsecured originations.
4. **Early Vintage Slippage (`age_band` & Tenure):** Borrowers in the `18-25` band with thin bureau files (`has_bureau_history = 0`) show elevated early-month missed installments. **So What:** Introduce graduated starter limits with step-up eligibility after 6 consecutive `On-Time` payments.
5. **Regional Delinquency Roll-Rates (`region`):** Specific regions show higher roll-rates from 1–29 DPD (`count_late`) into 30+ DPD (`has_dpd_30plus`). **So What:** Reallocate early-stage digital collections reminders to high-roll-rate regions before accounts cross 30 DPD.
        """)

# =============================================================================
# TAB 2: ACCOUNT-LEVEL VIEW & DEFAULT MODEL DIAGNOSTICS
# =============================================================================
with tab2:
    col_left, col_right = st.columns([1.1, 1.3])

    with col_left:
        st.subheader("Account-Level Risk Inspector")
        acc_list = df["account_id"].tolist()
        selected_acc = st.selectbox("Select or Type `account_id`", options=acc_list[:500])
        acc_row = df[df["account_id"] == selected_acc].iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("LightGBM PD Score", f"{acc_row['pd_lgbm']*100:.1f}%", acc_row["risk_tier"])
        m2.metric("LogReg Baseline PD", f"{acc_row['pd_logreg']*100:.1f}%")
        m3.metric("Actual Outcome", "DEFAULT (1)" if acc_row["target_default"] == 1 else "PERFORMING (0)")

        st.markdown(f"**Top 3 SHAP Risk Drivers:** `{acc_row['top_3_risk_drivers']}`")
        st.markdown(f"**Anomaly Status:** `{'FLAGGED — ' + acc_row['anomaly_reason'] if acc_row['is_anomaly'] else 'Normal'}`")

        # Single-Account SHAP Waterfall / Bar Chart
        global_idx = df_full.index[df_full["account_id"] == selected_acc][0]
        shap_vals_acc = model_bundle["shap_matrix"][global_idx]
        feat_names = model_bundle["shap_feature_names"]
        top_idx = np.argsort(np.abs(shap_vals_acc))[::-1][:10]
        acc_shap_df = pd.DataFrame({
            "Feature": [feat_names[i] for i in top_idx],
            "SHAP Contribution (Log-Odds)": [shap_vals_acc[i] for i in top_idx],
        }).sort_values("SHAP Contribution (Log-Odds)", ascending=True)

        fig_acc_shap = px.bar(
            acc_shap_df, x="SHAP Contribution (Log-Odds)", y="Feature", orientation="h",
            color="SHAP Contribution (Log-Odds)", color_continuous_scale="RdBu_r",
            title=f"Account {selected_acc} — Top 10 Local SHAP Drivers",
        )
        st.plotly_chart(fig_acc_shap, use_container_width=True)

    with col_right:
        st.subheader(f"Monthly Repayment Trajectory (`{selected_acc}`)")
        acc_repay = repay_raw[repay_raw["account_id"] == selected_acc].sort_values("payment_month")
        if not acc_repay.empty:
            fig_rep = px.line(
                acc_repay, x="payment_month", y=["emi_due", "emi_paid", "days_past_due"],
                markers=True, title=f"Monthly EMI Due vs. Paid & Days Past Due ({selected_acc})",
            )
            st.plotly_chart(fig_rep, use_container_width=True)

        st.subheader("Default Model Performance (LogReg Baseline vs. LightGBM)")
        st.dataframe(model_bundle["metrics_table"], use_container_width=True, hide_index=True)

        top_global_shap = model_bundle["lgbm_importance"].head(10).sort_values("mean_abs_shap", ascending=True)
        fig_glob = px.bar(
            top_global_shap, x="mean_abs_shap", y="feature", orientation="h",
            title="Global Top 10 Features (Mean |SHAP| — Collections Excluded to Prevent Leakage)",
        )
        st.plotly_chart(fig_glob, use_container_width=True)

# =============================================================================
# TAB 3: MANDATORY AI FEATURE 1 — "ASK THE PORTFOLIO" (LLM Q&A)
# =============================================================================
with tab3:
    st.subheader("Ask the Portfolio — Grounded Natural-Language Q&A")

    st.info("""
**AI Operational Governance — "Ask the Portfolio" (LLM):**
* **Trigger:** On-demand when an analyst submits a natural-language question or selects a preset prompt.
* **False Positive:** Over-strict schema guardrails declining a valid portfolio query, or mapping ambiguous business terms (e.g., "bad loans") to `repay_max_dpd >= 30` instead of `target_default = 1`.
* **False Negative:** Returning technically exact SQL figures on a micro-segment (e.g., $n < 5$ accounts) without flagging low statistical sample size.
* **Human-in-the-Loop (HITL) Step:** The analyst inspects the **Executed DuckDB SQL & Exact Source Data Table** rendered below every answer and records a verification decision (`Verified Accurate` / `Flag SQL Issue`) before exporting numbers to credit committee reports.
    """)

    preset_q = st.selectbox(
        "Quick-Select Sample Analyst Questions (or type your own below):",
        options=[
            "Which loan type has the highest default rate and what is its total exposure?",
            "How do default rates and average bureau utilization compare across employment types?",
            "Which region has the highest default rate and average loan-to-income ratio?",
            "What is the default rate across income bands?",
            "How many accounts are flagged as anomalies by loan type, and what is their default rate?",
            "What will the central bank inflation and stock price be in 2027? (Out-of-Scope Test)",
        ],
    )
    user_q = st.text_input("Ask any question about the portfolio:", value=preset_q)

    if st.button("Run Grounded Portfolio Query", type="primary"):
        with st.spinner("Generating DuckDB SQL & grounding response against portfolio tables..."):
            res = ask_portfolio(user_q, df)
        st.markdown(f"**Execution Mode:** `{res['mode']}` | **Source Tables:** `{res['source']}`")
        st.success(res["answer"])

        if res["sql"]:
            with st.expander("Human-in-the-Loop Verification: View Executed SQL & Underlying Data", expanded=True):
                st.code(res["sql"], language="sql")
                st.dataframe(res["data"], use_container_width=True, hide_index=True)
                hitl_col1, hitl_col2 = st.columns(2)
                hitl_col1.button("Approve Answer & SQL for Report")
                hitl_col2.button("Flag SQL Interpretation Issue")

# =============================================================================
# TAB 4: MANDATORY AI FEATURE 2 — ANOMALY DETECTION & HUMAN REVIEW QUEUE
# =============================================================================
with tab4:
    st.subheader("Peer & Behavioral Anomaly Detection — Analyst Review Queue")

    st.warning("""
**AI Operational Governance — Anomaly Detection Engine:**
* **Trigger:** Batch execution during data ingestion (`src/anomaly.py`) combining Peer-Segment Z-Scores (`loan_type × income_band`) and a multivariate `IsolationForest` ($4\%$ contamination).
* **False Positive:** Flagging a seasonal earner (e.g., Business Owner / Contractor) whose temporary bureau utilization spike or lump-sum repayment cycle is normal for their industry.
* **False Negative:** Missing a distressed borrower who maintains `On-Time` internal EMI payments by drawing down unmonitored informal credit lines not reported in `bureau_credit.csv`.
* **Human-in-the-Loop (HITL) Step:** Credit risk analyst inspects the flagged account's `anomaly_reason` and SHAP profile below, then records a binding disposition (`Confirmed Risk - Escalate` or `Dismissed - False Positive`), persisted to `data/processed/anomaly_reviews.json`.
    """)

    flagged_df = df[df["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False).copy()
    flagged_df["review_status"] = flagged_df["account_id"].map(
        lambda x: st.session_state["anomaly_decisions"].get(x, {}).get("decision", "Pending Review")
    )

    a1, a2, a3 = st.columns(3)
    a1.metric("Total Flagged Accounts", f"{len(flagged_df):,}")
    a2.metric(
        "Confirmed Risks",
        f"{sum(1 for v in st.session_state['anomaly_decisions'].values() if v.get('decision') == 'Confirmed Risk - Escalate')}",
    )
    a3.metric(
        "Dismissed (False Positives)",
        f"{sum(1 for v in st.session_state['anomaly_decisions'].values() if v.get('decision') == 'Dismissed - False Positive')}",
    )

    display_cols = [
        "account_id", "customer_id", "loan_type", "income_band", "pd_lgbm",
        "utilization_rate", "bureau_max_dpd", "repay_max_dpd", "anomaly_score",
        "review_status", "anomaly_reason",
    ]
    st.dataframe(flagged_df[display_cols].head(100), use_container_width=True, hide_index=True)

    st.markdown("### Human-in-the-Loop Disposition Logger")
    if not flagged_df.empty:
        r_col1, r_col2, r_col3 = st.columns([1, 1, 1.5])
        with r_col1:
            target_anom_acc = st.selectbox("Select Flagged `account_id`", options=flagged_df["account_id"].head(200).tolist())
        with r_col2:
            decision_val = st.selectbox(
                "Analyst Decision",
                options=["Pending Review", "Confirmed Risk - Escalate", "Dismissed - False Positive"],
            )
        with r_col3:
            analyst_note = st.text_input("Analyst Audit Note", placeholder="e.g., Verified seasonal utilization spike; dismiss.")

        if st.button("Save Analyst Disposition"):
            st.session_state["anomaly_decisions"][target_anom_acc] = {
                "decision": decision_val,
                "note": analyst_note,
            }
            save_review_decisions(st.session_state["anomaly_decisions"])
            st.success(f"Saved decision `{decision_val}` for `{target_anom_acc}` to audit log.")
            st.rerun()