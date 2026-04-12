from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


@dataclass
class ThresholdResult:
    threshold: float
    utility: float
    approval_rate: float
    default_rate_approved: float


def optimize_approval_threshold(
    scores: pd.Series,
    y: pd.Series,
    gain: float = 1.0,
    loss: float = 4.0,
) -> ThresholdResult:
    s = pd.to_numeric(scores, errors="coerce").fillna(scores.median())
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    if s.empty:
        return ThresholdResult(threshold=0.0, utility=0.0, approval_rate=0.0, default_rate_approved=0.0)

    grid = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 91)))
    if len(grid) == 0:
        grid = np.array([float(s.median())])

    best = ThresholdResult(threshold=float(grid[0]), utility=-1e18, approval_rate=0.0, default_rate_approved=0.0)
    for thr in grid:
        approved = s <= thr
        tp_non_default = int(((approved) & (y == 0)).sum())
        fp_default = int(((approved) & (y == 1)).sum())
        utility = tp_non_default * gain - fp_default * loss
        appr_rate = float(approved.mean())
        default_rate = float(y[approved].mean()) if approved.any() else 0.0

        if utility > best.utility:
            best = ThresholdResult(
                threshold=float(thr),
                utility=float(utility),
                approval_rate=appr_rate,
                default_rate_approved=default_rate,
            )
    return best


def build_industry_approval_policy(
    loan_risk_df: pd.DataFrame,
    industry_col: str,
    score_col: str,
    target_col: str,
    min_samples_per_industry: int = 6,
    gain: float = 1.0,
    loss: float = 4.0,
) -> pd.DataFrame:
    global_best = optimize_approval_threshold(
        loan_risk_df[score_col],
        loan_risk_df[target_col],
        gain=gain,
        loss=loss,
    )
    rows = []
    for industry, grp in loan_risk_df.groupby(industry_col):
        y = grp[target_col].astype(int)
        enough = len(grp) >= min_samples_per_industry and y.nunique() >= 2
        if enough:
            best = optimize_approval_threshold(grp[score_col], y, gain=gain, loss=loss)
            source = "industry"
        else:
            best = global_best
            source = "global_fallback"

        rows.append(
            {
                industry_col: industry,
                "threshold": best.threshold,
                "threshold_source": source,
                "sample_size": len(grp),
                "event_rate": float(y.mean()),
                "expected_utility": best.utility,
                "approval_rate_at_threshold": best.approval_rate,
                "default_rate_if_approved": best.default_rate_approved,
            }
        )

    return pd.DataFrame(rows).sort_values(by="expected_utility", ascending=False).reset_index(drop=True)


def apply_approval_policy(
    loan_risk_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    industry_col: str,
    score_col: str,
) -> pd.DataFrame:
    out = loan_risk_df.copy()
    map_df = policy_df[[industry_col, "threshold", "threshold_source"]].copy()
    out = out.merge(map_df, on=industry_col, how="left")
    global_thr = float(policy_df["threshold"].median()) if not policy_df.empty else float(out[score_col].median())
    out["threshold"] = out["threshold"].fillna(global_thr)
    out["threshold_source"] = out["threshold_source"].fillna("global_median")
    out["loan_decision"] = np.where(out[score_col] <= out["threshold"], "Approve", "Reject")
    return out


def optimize_warning_threshold(
    probs: pd.Series,
    y: pd.Series,
    min_positive_predictions: int = 1,
) -> Tuple[float, float]:
    p = pd.to_numeric(probs, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    grid = np.unique(np.quantile(p, np.linspace(0.1, 0.95, 86)))
    if len(grid) == 0:
        grid = np.array([0.2])

    best_thr, best_f1 = float(grid[0]), -1.0
    for thr in grid:
        pred = (p >= thr).astype(int)
        if pred.sum() < min_positive_predictions:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr, max(best_f1, 0.0)


def build_industry_warning_policy(
    warning_df: pd.DataFrame,
    industry_col: str,
    prob_col: str,
    target_col: str,
    min_samples_per_industry: int = 30,
) -> pd.DataFrame:
    global_thr, global_f1 = optimize_warning_threshold(warning_df[prob_col], warning_df[target_col])
    rows = []
    for industry, grp in warning_df.groupby(industry_col):
        y = grp[target_col].astype(int)
        enough = len(grp) >= min_samples_per_industry and y.sum() >= 2
        if enough:
            thr, f1 = optimize_warning_threshold(grp[prob_col], y)
            source = "industry"
        else:
            thr, f1 = global_thr, global_f1
            source = "global_fallback"
        rows.append(
            {
                industry_col: industry,
                "warning_threshold": float(thr),
                "warning_threshold_source": source,
                "sample_size": len(grp),
                "positive_rate": float(y.mean()),
                "f1_at_threshold": float(f1),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def apply_warning_policy(
    df: pd.DataFrame,
    policy_df: pd.DataFrame,
    industry_col: str,
    prob_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(
        policy_df[[industry_col, "warning_threshold", "warning_threshold_source"]],
        on=industry_col,
        how="left",
    )
    global_thr = float(policy_df["warning_threshold"].median()) if not policy_df.empty else 0.2
    out["warning_threshold"] = out["warning_threshold"].fillna(global_thr)
    out["warning_threshold_source"] = out["warning_threshold_source"].fillna("global_median")

    amber_ratio = 0.6
    out["risk_level"] = np.where(
        out[prob_col] >= out["warning_threshold"],
        "RED",
        np.where(out[prob_col] >= out["warning_threshold"] * amber_ratio, "AMBER", "GREEN"),
    )
    out["warning_flag"] = (out["risk_level"].isin(["RED", "AMBER"])).astype(int)
    return out

