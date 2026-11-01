# src/anomaly.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_portfolio_anomalies(scored_master: pd.DataFrame, contamination: float = 0.04) -> pd.DataFrame:
    """Flag accounts with unusual repayment or bureau patterns vs. their peer segment & own history."""
    df = scored_master.copy()

    # Define peer group: loan_type x income_band
    peer_cols = [c for c in ["loan_type", "income_band"] if c in df.columns]
    if not peer_cols:
        df["_peer_group"] = "All"
        peer_cols = ["_peer_group"]

    # Compute peer-group Z-scores for key behavioral and bureau metrics
    z_metrics = [
        ("utilization_rate", "z_peer_bureau_util"),
        ("loan_to_income_ratio", "z_peer_lti"),
        ("repay_max_dpd", "z_peer_repay_dpd"),
        ("bureau_max_dpd", "z_peer_bureau_dpd"),
        ("dpd_trend_recent_vs_early", "z_peer_dpd_spike"),
    ]

    for col, z_name in z_metrics:
        if col in df.columns:
            grp_mean = df.groupby(peer_cols)[col].transform("mean")
            grp_std = df.groupby(peer_cols)[col].transform("std").replace(0, 1.0).fillna(1.0)
            df[z_name] = ((df[col] - grp_mean) / grp_std).fillna(0.0)
        else:
            df[z_name] = 0.0

    # Multivariate Isolation Forest on behavioral/bureau features
    iso_features = [
        c for c in [
            "utilization_rate", "max_line_utilization", "bureau_max_dpd",
            "repayment_ratio", "repay_max_dpd", "dpd_trend_recent_vs_early",
            "loan_to_income_ratio", "actual_dti_ratio",
        ] if c in df.columns
    ]
    X_iso = StandardScaler().fit_transform(df[iso_features].fillna(0.0))
    iso = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
    df["iso_flag"] = (iso.fit_predict(X_iso) == -1).astype(int)
    df["anomaly_score"] = (-iso.decision_function(X_iso)).round(4)

    # Explicit Domain Archetypes (Explainable Rules + Multivariate Outliers)
    # Archetype 1: External Bureau Contagion (Paying internal loan on time, but severe external bureau distress)
    cond_contagion = (
        (df.get("repay_max_dpd", 0) == 0)
        & ((df.get("bureau_max_dpd", 0) >= 45) | (df["z_peer_bureau_util"] >= 2.5))
    )

    # Archetype 2: Sudden Repayment Shock (Historically clean early months, sharp spike in recent 3m DPD)
    cond_shock = (
        (df.get("early_3m_max_dpd", 0) == 0)
        & (df.get("recent_3m_max_dpd", 0) >= 30)
        & (df["z_peer_dpd_spike"] >= 2.0)
    )

    # Archetype 3: Extreme Peer Leverage Outlier (LTI or DTI > 3 SD above peer group mean)
    cond_leverage = (df["z_peer_lti"] >= 3.0) & (df.get("utilization_rate", 0) >= 0.80)

    # Archetype 4: Uncollected Severe Delinquency (Operational Gap: 60+ internal DPD with 0 collection events)
    cond_ops_gap = (df.get("repay_max_dpd", 0) >= 60) & (df.get("coll_event_cnt", 0) == 0)

    df["is_anomaly"] = (cond_contagion | cond_shock | cond_leverage | cond_ops_gap | (df["iso_flag"] == 1)).astype(int)

    # Build human-readable explanation for every flagged account
    reasons = []
    for idx, row in df.iterrows():
        if not row["is_anomaly"]:
            reasons.append("Normal")
            continue
        r_list = []
        if cond_contagion.loc[idx]:
            r_list.append(
                f"Silent Bureau Contagion: 0 internal DPD but external bureau DPD={row.get('bureau_max_dpd', 0):.0f}d "
                f"& util={row.get('utilization_rate', 0)*100:.1f}% ({row['z_peer_bureau_util']:+.1f} SD vs peer)"
            )
        if cond_shock.loc[idx]:
            r_list.append(
                f"Sudden Payment Shock: 0 early DPD -> {row.get('recent_3m_max_dpd', 0):.0f}d recent 3m DPD "
                f"({row['z_peer_dpd_spike']:+.1f} SD trajectory spike vs peer)"
            )
        if cond_leverage.loc[idx]:
            r_list.append(
                f"Peer Leverage Outlier: LTI={row.get('loan_to_income_ratio', 0):.2f}x "
                f"({row['z_peer_lti']:+.1f} SD above {row.get('loan_type', '')} / {row.get('income_band', '')} peers)"
            )
        if cond_ops_gap.loc[idx]:
            r_list.append(
                f"Collections Coverage Gap: Internal max DPD={row.get('repay_max_dpd', 0):.0f}d with 0 recorded collection actions"
            )
        if not r_list:
            r_list.append(
                f"Multivariate Behavioral Outlier (Isolation Forest score={row['anomaly_score']:.3f}; "
                f"Util={row.get('utilization_rate', 0)*100:.1f}%, Repay Ratio={row.get('repayment_ratio', 1)*100:.1f}%)"
            )
        reasons.append(" ; ".join(r_list))

    df["anomaly_reason"] = reasons
    return df