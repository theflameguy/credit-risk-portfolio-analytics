from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw"


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """1. Clean customers.csv by imputing missing values and standardizing labels."""
    df = df.copy()

    # Standardize gender column
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.title()
        gender_map = {
            'M': 'Male',
            'F': 'Female',
            'Male': 'Male',
            'Female': 'Female'
        }
        df['gender'] = df['gender'].map(gender_map).fillna('Unknown')

    # The 3 columns with missing values based on the data info:
    cols_to_impute = ['years_employed', 'num_dependents', 'years_at_residence']
    valid_cols = [c for c in cols_to_impute if c in df.columns]

    if valid_cols:
        # Mark 1 if any value is missing (to be imputed) across the selected columns, else 0
        df['cust_imputed_flag'] = df[valid_cols].isnull().any(axis=1).astype(int)

        imputer = KNNImputer(n_neighbors=5)
        df[valid_cols] = imputer.fit_transform(df[valid_cols])
    else:
        df['cust_imputed_flag'] = 0

    return df

def clean_loan_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Clean loan_accounts.csv by only keeping the interest_rate_pct KNN imputation and the acc_imputed_val flag."""
    df = df.copy()
    rate_col = "interest_rate_pct"

    if rate_col in df.columns:
        df["interest_rate_pct"] = pd.to_numeric(df[rate_col], errors="coerce")
        df["acc_imputed_flag"] = df["interest_rate_pct"].isnull().astype(int)

        imputer = KNNImputer(n_neighbors=5)
        fit_cols = [c for c in ["loan_amount", "interest_rate_pct"] if c in df.columns]
        if "interest_rate_pct" in fit_cols:
            df[fit_cols] = imputer.fit_transform(df[fit_cols])
    else:
        df["acc_imputed_flag"] = 0


    return df

def aggregate_bureau(df: pd.DataFrame) -> pd.DataFrame:
    """3. Aggregate bureau_credit.csv (23k rows) -> 1 row per customer_id with lender, type, status & dates."""
    df = df.copy()

    # 1. Numeric conversion & multi-column KNN imputation for days_past_due
    df["days_past_due"] = pd.to_numeric(df["days_past_due"], errors="coerce")
    df["bur_imputed_flag"] = df["days_past_due"].isnull().astype(int)

    fit_cols = [c for c in ["sanctioned_amount", "outstanding_amount", "days_past_due"] if c in df.columns]
    imputer = KNNImputer(n_neighbors=5)
    df[fit_cols] = imputer.fit_transform(df[fit_cols])
    df["days_past_due"] = np.where(df["days_past_due"] < 1.0, 0.0, df["days_past_due"])

    # Line-level utilization (catches single maxed-out credit lines masked by larger loans)
    df["line_util"] = np.where(
        df["sanctioned_amount"] > 0,
        (df["outstanding_amount"] / df["sanctioned_amount"]).clip(0.0, 3.0),
        0.0,
    )

    # Parse dates to capture credit history depth and recent credit hunger
    df["date_opened"] = pd.to_datetime(df["date_opened"], errors="coerce")
    df["date_closed"] = pd.to_datetime(df["date_closed"], errors="coerce")

    # 2. Core numeric aggregations per customer
    bureau_agg = (
        df.groupby("customer_id")
        .agg(
            total_accounts=("bureau_id", "count"),
            total_sanctioned=("sanctioned_amount", "sum"),
            total_outstanding=("outstanding_amount", "sum"),
            max_dpd=("days_past_due", "max"),
            avg_bureau_dpd=("days_past_due", "mean"),
            std_bureau_dpd=("days_past_due", "std"),
            count_overdue_loans=("days_past_due", lambda x: (x > 0).sum()),
            count_dpd_30plus_loans=("days_past_due", lambda x: (x >= 30).sum()),
            count_dpd_60plus_loans=("days_past_due", lambda x: (x >= 60).sum()),
            max_line_utilization=("line_util", "max"),
            avg_line_utilization=("line_util", "mean"),
            earliest_bureau_open=("date_opened", "min"),
            latest_bureau_open=("date_opened", "max"),
            closed_accounts_cnt=("date_closed", lambda x: x.notna().sum()),
            bur_imputed_flag=("bur_imputed_flag", "max"),
        )
        .reset_index()
    )
    bureau_agg["std_bureau_dpd"] = bureau_agg["std_bureau_dpd"].fillna(0.0)

    # Credit history span in days (between earliest and latest external credit line)
    bureau_agg["bureau_history_span_days"] = (
        bureau_agg["latest_bureau_open"] - bureau_agg["earliest_bureau_open"]
    ).dt.days.fillna(0.0).clip(lower=0.0)

    bureau_agg["utilization_rate"] = np.where(
        bureau_agg["total_sanctioned"] > 0,
        (bureau_agg["total_outstanding"] / bureau_agg["total_sanctioned"]).clip(0.0, 3.0),
        0.0,
    )
    bureau_agg["overdue_account_share"] = (
        bureau_agg["count_overdue_loans"] / bureau_agg["total_accounts"].clip(lower=1)
    )

    # 3. Pivot account_status, lender_category, and credit_type (counts + proportions)
    status_ct = (
        pd.crosstab(df["customer_id"], df["account_status"])
        .add_prefix("bur_status_cnt_")
        .reset_index()
    )
    status_share = (
        pd.crosstab(df["customer_id"], df["account_status"], normalize="index")
        .add_prefix("bur_status_pct_")
        .reset_index()
    )
    lender_ct = (
        pd.crosstab(df["customer_id"], df["lender_category"])
        .add_prefix("bur_lender_cnt_")
        .reset_index()
    )
    lender_share = (
        pd.crosstab(df["customer_id"], df["lender_category"], normalize="index")
        .add_prefix("bur_lender_pct_")
        .reset_index()
    )
    type_ct = (
        pd.crosstab(df["customer_id"], df["credit_type"])
        .add_prefix("bur_type_cnt_")
        .reset_index()
    )

    # Keep backwards-compatible active_accounts column
    bureau_agg = (
        bureau_agg.merge(status_ct, on="customer_id", how="left")
        .merge(status_share, on="customer_id", how="left")
        .merge(lender_ct, on="customer_id", how="left")
        .merge(lender_share, on="customer_id", how="left")
        .merge(type_ct, on="customer_id", how="left")
    )
    bureau_agg["active_accounts"] = bureau_agg.get("bur_status_cnt_Active", 0)

    return bureau_agg


def aggregate_repayments(df: pd.DataFrame) -> pd.DataFrame:
    """4. Aggregate repayment_history.csv (176k rows) -> 1 row per account_id."""
    repayment_hist = df.copy()
    repayment_hist["payment_month"] = pd.to_numeric(repayment_hist["payment_month"], errors="coerce")
    repayment_hist = repayment_hist.sort_values(by=["account_id", "payment_month"]).copy()

    # Monthly shortfall at installment grain
    repayment_hist["month_shortfall"] = (repayment_hist["emi_due"] - repayment_hist["emi_paid"]).clip(lower=0.0)
    repayment_hist["month_pay_ratio"] = np.where(
        repayment_hist["emi_due"] > 0,
        repayment_hist["emi_paid"] / repayment_hist["emi_due"],
        1.0,
    )

    # 1. Lifetime aggregates per account
    agg_lifetime = (
        repayment_hist.groupby("account_id")
        .agg(
            months_observed=("payment_month", "count"),
            max_dpd=("days_past_due", "max"),
            avg_dpd=("days_past_due", "mean"),
            std_dpd=("days_past_due", "std"),
            latest_dpd=("days_past_due", "last"),
            first_dpd=("days_past_due", "first"),
            count_dpd_gt0=("days_past_due", lambda s: (s > 0).sum()),
            count_dpd_30plus=("days_past_due", lambda s: (s >= 30).sum()),
            count_dpd_60plus=("days_past_due", lambda s: (s >= 60).sum()),
            count_late=("payment_status", lambda s: (s == "Late").sum()),
            count_missed=("payment_status", lambda s: (s == "Missed").sum()),
            total_emi_due=("emi_due", "sum"),
            total_emi_paid=("emi_paid", "sum"),
            mean_emi_due=("emi_due", "mean"),
            max_month_shortfall=("month_shortfall", "max"),
            min_month_pay_ratio=("month_pay_ratio", "min"),
            initial_balance=("outstanding_balance", "first"),
            latest_balance=("outstanding_balance", "last"),
        )
        .reset_index()
    )
    agg_lifetime["std_dpd"] = agg_lifetime["std_dpd"].fillna(0.0)

    agg_lifetime["repayment_ratio"] = np.where(
        agg_lifetime["total_emi_due"] > 0,
        agg_lifetime["total_emi_paid"] / agg_lifetime["total_emi_due"],
        1.0,
    )
    agg_lifetime["total_shortfall"] = agg_lifetime["total_emi_due"] - agg_lifetime["total_emi_paid"]
    agg_lifetime["balance_reduction_pct"] = np.where(
        agg_lifetime["initial_balance"] > 0,
        (agg_lifetime["initial_balance"] - agg_lifetime["latest_balance"]) / agg_lifetime["initial_balance"],
        0.0,
    )
    agg_lifetime["has_dpd_30plus"] = (agg_lifetime["max_dpd"] >= 30).astype(int)
    agg_lifetime["dpd_month_share"] = agg_lifetime["count_dpd_gt0"] / agg_lifetime["months_observed"].clip(lower=1)

    # 2. Pivot all payment_status categories (counts and shares)
    pay_status_ct = (
        pd.crosstab(repayment_hist["account_id"], repayment_hist["payment_status"])
        .add_prefix("rep_status_cnt_")
        .reset_index()
    )
    pay_status_share = (
        pd.crosstab(repayment_hist["account_id"], repayment_hist["payment_status"], normalize="index")
        .add_prefix("rep_status_pct_")
        .reset_index()
    )

    # 3. Recency window (Last 3 installments) & Early window (First 3 installments)
    repayment_hist["month_rank_desc"] = repayment_hist.groupby("account_id")["payment_month"].rank(
        ascending=False, method="first"
    )
    repayment_hist["month_rank_asc"] = repayment_hist.groupby("account_id")["payment_month"].rank(
        ascending=True, method="first"
    )

    agg_recent = (
        repayment_hist[repayment_hist["month_rank_desc"] <= 3]
        .groupby("account_id")
        .agg(
            recent_3m_max_dpd=("days_past_due", "max"),
            recent_3m_avg_dpd=("days_past_due", "mean"),
            recent_3m_late_count=("payment_status", lambda s: (s == "Late").sum()),
            recent_3m_missed_count=("payment_status", lambda s: (s == "Missed").sum()),
            recent_3m_pay_ratio=("month_pay_ratio", "mean"),
        )
        .reset_index()
    )

    agg_early = (
        repayment_hist[repayment_hist["month_rank_asc"] <= 3]
        .groupby("account_id")
        .agg(
            early_3m_max_dpd=("days_past_due", "max"),
            early_3m_avg_dpd=("days_past_due", "mean"),
            early_3m_pay_ratio=("month_pay_ratio", "mean"),
        )
        .reset_index()
    )

    repayment_features = (
        agg_lifetime.merge(pay_status_ct, on="account_id", how="left")
        .merge(pay_status_share, on="account_id", how="left")
        .merge(agg_recent, on="account_id", how="left")
        .merge(agg_early, on="account_id", how="left")
    )

    # Trajectory features: is borrower deteriorating in recent months compared to early months?
    repayment_features["dpd_trend_recent_vs_early"] = (
        repayment_features["recent_3m_avg_dpd"] - repayment_features["early_3m_avg_dpd"]
    )
    repayment_features["pay_ratio_trend_recent_vs_early"] = (
        repayment_features["recent_3m_pay_ratio"] - repayment_features["early_3m_pay_ratio"]
    )

    return repayment_features

def aggregate_collections(df: pd.DataFrame) -> pd.DataFrame:
    """5. Aggregate collections_events.csv (8.6k rows) -> 1 row per account_id.
    NOTE: All columns prefixed with `coll_` to isolate post-delinquency actions from PD training.
    """
    collection_events = df.copy()
    # 1. Parse dates and sort chronologically
    collection_events["event_date"] = pd.to_datetime(
        collection_events["event_date"]
    )
    collection_events = collection_events.sort_values(
        by=["account_id", "event_date"]
    ).copy()

    # 2. Pivot action types and outcomes to get counts per account
    action_counts = (
        pd.crosstab(
            collection_events["account_id"], collection_events["action_type"]
        )
        .add_prefix("count_action_")
        .reset_index()
    )

    outcome_counts = (
        pd.crosstab(collection_events["account_id"], collection_events["outcome"])
        .add_prefix("count_outcome_")
        .reset_index()
    )

    # 3. Aggregate totals, recovery amounts, and internal time span
    event_summary = (
        collection_events.groupby("account_id")
        .agg(
            total_collection_events=("collection_event_id", "count"),
            total_amount_recovered=("amount_recovered", "sum"),
            max_amount_recovered=("amount_recovered", "max"),
            first_event_date=("event_date", "min"),
            last_event_date=("event_date", "max"),
            # latest_action=("action_type", "last"),
            # latest_outcome=("outcome", "last"),
        )
        .reset_index()
    )

    # Internal duration in days (no system date used)
    event_summary["collection_span_days"] = (
        event_summary["last_event_date"] - event_summary["first_event_date"]
    ).dt.days

    # Drop intermediate date columns if not required downstream
    event_summary = event_summary.drop(
        columns=["first_event_date", "last_event_date"]
    )

    # 4. Merge all engineered collection components into 1 row per account
    coll_features = event_summary.merge(
        action_counts, on="account_id", how="left"
    ).merge(outcome_counts, on="account_id", how="left")

    # 5. Derived behavioral ratios & severity flags
    total_events = coll_features["total_collection_events"]

    coll_features["dispute_rate"] = (
        coll_features.get("count_outcome_Disputed", 0) / total_events
    )
    coll_features["escalation_rate"] = (
        coll_features.get("count_outcome_Escalated", 0) / total_events
    )
    coll_features["positive_response_rate"] = (
        coll_features.get("count_outcome_Payment Received", 0)
        + coll_features.get("count_outcome_Promise to Pay", 0)
    ) / total_events

    coll_features["has_legal_action"] = (
        coll_features.get("count_action_Legal Notice", 0) > 0
    ).astype(int)
    coll_features["has_writeoff_referral"] = (
        coll_features.get("count_action_Written-Off Referral", 0) > 0
    ).astype(int)
    coll_features["has_recovered_funds"] = (
        coll_features["total_amount_recovered"] > 0
    ).astype(int)

    return coll_features

def build_master_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:

    customers_raw = pd.read_csv(raw_dir / "customers.csv")
    loans_raw = pd.read_csv(raw_dir / "loan_accounts.csv")
    bureau_raw = pd.read_csv(raw_dir / "bureau_credit.csv")
    repay_raw = pd.read_csv(raw_dir / "repayment_history.csv")
    coll_raw = pd.read_csv(raw_dir / "collections_events.csv")

    customers = clean_customers(customers_raw)
    loans = clean_loan_accounts(loans_raw)
    bureau_agg = aggregate_bureau(bureau_raw)
    repay_agg = aggregate_repayments(repay_raw)
    coll_agg = aggregate_collections(coll_raw)

    # Resolve column name conflicts before merge
    if "max_dpd" in bureau_agg.columns:
        bureau_agg = bureau_agg.rename(columns={"max_dpd": "bureau_max_dpd"})
    if "max_dpd" in repay_agg.columns:
        repay_agg = repay_agg.rename(columns={"max_dpd": "repay_max_dpd"})

    # Left join onto loan_accounts spine
    master = (
        loans.merge(customers, on="customer_id", how="left")
        .merge(bureau_agg, on="customer_id", how="left")
        .merge(repay_agg, on="account_id", how="left")
        .merge(coll_agg, on="account_id", how="left")
    )

    # Cross-table velocity & multi-loan features
    orig_dt = pd.to_datetime(master["origination_date"], errors="coerce")
    since_dt = pd.to_datetime(master["customer_since"], errors="coerce")
    master["relationship_years"] = ((orig_dt - since_dt).dt.days / 365.25).clip(lower=0.0).fillna(0.0)

    if "latest_bureau_open" in master.columns:
        master["days_from_last_bureau_to_orig"] = (orig_dt - master["latest_bureau_open"]).dt.days.fillna(999.0)
        master = master.drop(columns=["earliest_bureau_open", "latest_bureau_open"])

    master["cust_internal_loan_cnt"] = master.groupby("customer_id")["account_id"].transform("count")
    master["cust_internal_total_debt"] = master.groupby("customer_id")["loan_amount"].transform("sum")

    if {"mean_emi_due", "annual_income"}.issubset(master.columns):
        master["actual_dti_ratio"] = (
            (master["mean_emi_due"].fillna(0) * 12.0) / master["annual_income"].clip(lower=1.0)
        ).clip(0.0, 5.0)

    # -------------------------------------------------------------------------
    # POST-JOIN MISSINGNESS HANDLING (0 bureau lines, 0 repayments, 0 collections)
    # -------------------------------------------------------------------------
    # 1. Bureau: flag presence, then fill missing counts/balances/rates with 0
    master["has_bureau_history"] = master["total_accounts"].notna().astype(int)
    bureau_fill_zero = [
        c for c in bureau_agg.columns if c != "customer_id" and c in master.columns
    ]
    master[bureau_fill_zero] = master[bureau_fill_zero].fillna(0)

    # 2. Repayments: if an account has no repayment rows, default ratio to 1.0 and counts to 0
    master["has_repayment_history"] = master["months_observed"].notna().astype(int)
    if "repayment_ratio" in master.columns:
        master["repayment_ratio"] = master["repayment_ratio"].fillna(1.0)
    repay_fill_zero = [
        c for c in repay_agg.columns if c not in ("account_id", "repayment_ratio")
    ]
    master[repay_fill_zero] = master[repay_fill_zero].fillna(0)

    # 3. Collections: flag presence, then fill all missing action/outcome counts & rates with 0
    master["has_collection_events"] = (
        master["total_collection_events"].notna().astype(int)
    )
    coll_fill_zero = [c for c in coll_agg.columns if c != "account_id"]
    master[coll_fill_zero] = master[coll_fill_zero].fillna(0)

    # Convert integer count/flag columns cleanly from float back to int
    int_Shared_Cols = [
        "total_accounts", "active_accounts", "count_overdue_loans", "bur_imputed_flag",
        "months_observed", "count_late", "count_missed", "has_dpd_30plus",
        "recent_3m_late_count", "recent_3m_missed_count", "total_collection_events",
        "has_legal_action", "has_writeoff_referral", "has_recovered_funds",
    ] + [c for c in coll_fill_zero if c.startswith(("count_action_", "count_outcome_"))]

    for col in int_Shared_Cols:
        if col in master.columns:
            master[col] = master[col].astype(int)


    # -------------------------------------------------------------------------
    # CROSS-TABLE RATIOS & PORTFOLIO ANALYTICS BANDS
    # -------------------------------------------------------------------------

    if "loan_amount" in master.columns and "annual_income" in master.columns:
        master["loan_to_income_ratio"] = (
            master["loan_amount"] / master["annual_income"].clip(lower=1.0)
        ).clip(0.0, 20.0)

    # Create analytical bands for the 5 required Portfolio Profiling dimensions
    if "annual_income" in master.columns:
        master["income_band"] = pd.qcut(
            master["annual_income"].rank(method="first"),
            q=4,
            labels=["Q1: Low Income", "Q2: Lower-Mid", "Q3: Upper-Mid", "Q4: High Income"],
        ).astype(str)

    if "age" in master.columns:
        master["age_band"] = pd.cut(
            master["age"],
            bins=[0, 25, 35, 45, 60, 120],
            labels=["18-25", "26-35", "36-45", "46-60", "60+"],
        ).astype(str)

    return master
