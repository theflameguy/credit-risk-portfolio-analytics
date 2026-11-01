# src/model.py
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
import shap


def select_model_features(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Dynamically include pre-default features and strictly block collections_events."""
    df = master.copy()

    blocked_exact = {
        "account_id", "customer_id", "target_default",
        "origination_date", "customer_since",
        "income_band", "age_band",
    }
    blocked_prefixes = ("coll_",)

    candidate_cols = [
        c for c in df.columns
        if c not in blocked_exact and not c.startswith(blocked_prefixes)
    ]

    cat_cols = [
        c for c in candidate_cols
        if (df[c].dtype == "object" or isinstance(df[c].dtype, pd.CategoricalDtype))
        and df[c].nunique(dropna=True) <= 30
    ]
    num_cols = [
        c for c in candidate_cols
        if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[num_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown")

    y = df["target_default"].astype(int).copy()
    return X, y, num_cols, cat_cols


def compute_credit_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute AUC, Gini (2*AUC - 1), and KS statistic."""
    auc = roc_auc_score(y_true, y_prob)
    gini = 2.0 * auc - 1.0
    ks_stat, _ = ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])
    return {
        "AUC": round(float(auc), 4),
        "Gini": round(float(gini), 4),
        "KS": round(float(ks_stat), 4),
    }


def train_default_models(master: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> dict:
    """Train Logistic Regression and Early-Stopped LightGBM models."""
    X, y, num_cols, cat_cols = select_model_features(master)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", min_frequency=30, sparse_output=False),
                cat_cols,
            ),
        ]
    )

    X_tr_mat = preprocessor.fit_transform(X_train)
    X_te_mat = preprocessor.transform(X_test)
    X_all_mat = preprocessor.transform(X)
    feat_names = [f.replace("num__", "").replace("cat__", "") for f in preprocessor.get_feature_names_out()]

    # 1. Logistic Regression Baseline
    logreg_clf = LogisticRegression(class_weight="balanced", C=0.05, max_iter=1000, random_state=random_state)
    logreg_clf.fit(X_tr_mat, y_train)

    logreg_train_prob = logreg_clf.predict_proba(X_tr_mat)[:, 1]
    logreg_test_prob = logreg_clf.predict_proba(X_te_mat)[:, 1]

    # 2. LightGBM with Early Stopping
    X_sub_tr, X_val, y_sub_tr, y_val = train_test_split(
        X_tr_mat, y_train, test_size=0.15, stratify=y_train, random_state=random_state
    )

    lgbm_clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        min_child_samples=150,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=5.0,
        reg_lambda=20.0,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )
    lgbm_clf.fit(
        X_sub_tr,
        y_sub_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    lgbm_train_prob = lgbm_clf.predict_proba(X_tr_mat)[:, 1]
    lgbm_test_prob = lgbm_clf.predict_proba(X_te_mat)[:, 1]

    # 3. Metrics Summary Table
    metrics_df = pd.DataFrame([
        {"Model": "Logistic Regression (Train)", **compute_credit_metrics(y_train.values, logreg_train_prob)},
        {"Model": "Logistic Regression (Test)", **compute_credit_metrics(y_test.values, logreg_test_prob)},
        {"Model": "LightGBM (Train)", **compute_credit_metrics(y_train.values, lgbm_train_prob)},
        {"Model": "LightGBM (Test)", **compute_credit_metrics(y_test.values, lgbm_test_prob)},
    ])

    # 4. Feature Importances & SHAP
    logreg_importance_df = (
        pd.DataFrame({
            "feature": feat_names,
            "coefficient": logreg_clf.coef_[0],
            "abs_importance": np.abs(logreg_clf.coef_[0]),
        })
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )

    explainer = shap.TreeExplainer(lgbm_clf)
    shap_values = explainer.shap_values(X_all_mat)
    shap_matrix = shap_values[1] if isinstance(shap_values, list) else (
        shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
    )

    lgbm_importance_df = (
        pd.DataFrame({
            "feature": feat_names,
            "lgbm_split_importance": lgbm_clf.feature_importances_,
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # 5. Scored Master Output for Dashboard
    scored_master = master.copy()
    scored_master["pd_logreg"] = logreg_clf.predict_proba(X_all_mat)[:, 1].round(4)
    scored_master["pd_lgbm"] = lgbm_clf.predict_proba(X_all_mat)[:, 1].round(4)

    scored_master["risk_tier"] = pd.cut(
        scored_master["pd_lgbm"],
        bins=[-0.01, 0.35, 0.50, 0.65, 1.01],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
    ).astype(str)

    feat_array = np.array(feat_names)
    top3_idx = np.argsort(-shap_matrix, axis=1)[:, :3]
    scored_master["top_3_risk_drivers"] = [
        " | ".join(f"{feat_array[j]} ({shap_matrix[row_i, j]:+.2f})" for j in idxs)
        for row_i, idxs in enumerate(top3_idx)
    ]

    return {
        "metrics_table": metrics_df,
        "logreg_importance": logreg_importance_df,
        "lgbm_importance": lgbm_importance_df,
        "scored_master": scored_master,
        "shap_matrix": shap_matrix,
        "shap_feature_names": feat_names,
    }