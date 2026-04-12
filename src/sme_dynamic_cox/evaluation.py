from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_time_split(
    loan_master: pd.DataFrame,
    test_size: float = 0.3,
    min_train_events: int = 4,
    min_test_events: int = 2,
) -> Tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    candidates = [0.6, 0.65, 0.7, 0.75, 0.8]
    preferred = max(0.5, min(0.85, 1.0 - test_size))
    candidates = [preferred] + [c for c in candidates if abs(c - preferred) > 1e-6]

    lm = loan_master.sort_values("loan_start_date").copy()
    for q in candidates:
        cutoff = lm["loan_start_date"].quantile(q)
        train = lm[lm["loan_start_date"] <= cutoff].copy()
        test = lm[lm["loan_start_date"] > cutoff].copy()
        if train.empty or test.empty:
            continue
        if train["event"].sum() >= min_train_events and test["event"].sum() >= min_test_events:
            return cutoff, train, test

    # Fallback: median split if constrained by very few events.
    cutoff = lm["loan_start_date"].median()
    train = lm[lm["loan_start_date"] <= cutoff].copy()
    test = lm[lm["loan_start_date"] > cutoff].copy()
    return cutoff, train, test


def build_origination_view(interval_data: pd.DataFrame, loan_master: pd.DataFrame) -> pd.DataFrame:
    first_rows = interval_data.sort_values(["loan_key", "start"]).groupby("loan_key", as_index=False).first()
    out = first_rows.copy()
    if "loan_event" in out.columns:
        out["event"] = pd.to_numeric(out["loan_event"], errors="coerce").fillna(out["event"]).astype(int)
    if "duration_days_loan" in out.columns:
        out["duration_days"] = pd.to_numeric(out["duration_days_loan"], errors="coerce").fillna(
            pd.to_numeric(out.get("duration_days"), errors="coerce")
        )

    # Fallback merge only if required fields are missing.
    if "event" not in out.columns or out["event"].isna().any() or "duration_days" not in out.columns:
        fallback = loan_master[["loan_key", "event", "duration_days", "industry_group", "loan_start_date"]].copy()
        out = out.merge(
            fallback,
            on=["loan_key", "industry_group", "loan_start_date"],
            how="left",
            suffixes=("", "_fallback"),
        )
        if "event_fallback" in out.columns:
            out["event"] = pd.to_numeric(out["event"], errors="coerce").fillna(out["event_fallback"]).astype(int)
            out = out.drop(columns=["event_fallback"])
        if "duration_days_fallback" in out.columns:
            out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce").fillna(
                out["duration_days_fallback"]
            )
            out = out.drop(columns=["duration_days_fallback"])

    out["event"] = pd.to_numeric(out["event"], errors="coerce").fillna(0).astype(int)
    out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce").fillna(1).clip(lower=1)
    return out


def add_future_event_label(interval_data: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    df = interval_data.copy()
    df["loan_event"] = pd.to_numeric(df["loan_event"], errors="coerce").fillna(0).astype(int)
    df["loan_event_time"] = pd.to_numeric(df["loan_event_time"], errors="coerce")
    df["stop"] = pd.to_numeric(df["stop"], errors="coerce")

    df["future_event_label"] = (
        (df["loan_event"] == 1)
        & (df["loan_event_time"] > df["stop"])
        & (df["loan_event_time"] <= df["stop"] + horizon_days)
    ).astype(int)

    df["monitoring_row"] = (
        (df["event"] == 0)
        & (df["stop"] < pd.to_numeric(df["duration_days"], errors="coerce").fillna(df["stop"]))
    ).astype(int)
    return df


def classification_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float) -> Dict[str, float]:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    y_score = pd.to_numeric(y_score, errors="coerce").fillna(0.0)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "avg_precision": float(average_precision_score(y_true, y_score)) if y_true.nunique() > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else float("nan"),
    }
    return metrics
