from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelArtifacts:
    model: CoxTimeVaryingFitter
    feature_cols: List[str]
    feature_medians: pd.Series
    scaler: StandardScaler
    strata_col: str


class DynamicCoxRiskModel:
    def __init__(self, penalizer: float = 0.2, l1_ratio: float = 0.0, strata_col: str = "industry_group") -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.strata_col = strata_col
        self.artifacts: ModelArtifacts | None = None
        self._baseline_by_strata: Dict[str, pd.Series] = {}

    def _prepare_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: Iterable[str],
        feature_medians: pd.Series | None = None,
        scaler: StandardScaler | None = None,
        fit: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
        X = df[list(feature_cols)].copy()
        if fit:
            medians = X.median(numeric_only=True)
        else:
            if feature_medians is None:
                raise ValueError("feature_medians must be provided when fit=False.")
            medians = feature_medians
        X = X.fillna(medians)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
        else:
            if scaler is None:
                raise ValueError("scaler must be provided when fit=False.")
            X_scaled = scaler.transform(X.values)

        X_df = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
        return X_df, medians, scaler

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> None:
        X_df, medians, scaler = self._prepare_feature_matrix(train_df, feature_cols, fit=True)
        fit_df = pd.concat(
            [
                train_df[["loan_key", "start", "stop", "event", self.strata_col]].reset_index(drop=True),
                X_df.reset_index(drop=True),
            ],
            axis=1,
        )

        ctv = CoxTimeVaryingFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            ctv.fit(
                fit_df,
                id_col="loan_key",
                event_col="event",
                start_col="start",
                stop_col="stop",
                strata=[self.strata_col],
                show_progress=False,
            )

        self.artifacts = ModelArtifacts(
            model=ctv,
            feature_cols=feature_cols,
            feature_medians=medians,
            scaler=scaler,
            strata_col=self.strata_col,
        )
        self._cache_baseline()

    def _cache_baseline(self) -> None:
        if self.artifacts is None:
            return
        base = self.artifacts.model.baseline_cumulative_hazard_
        self._baseline_by_strata = {}

        if base.empty:
            return

        cols = list(base.columns)
        for col in cols:
            key = str(col)
            series = base[col].dropna().astype(float)
            self._baseline_by_strata[key] = series

    def _get_artifacts(self) -> ModelArtifacts:
        if self.artifacts is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.artifacts

    def _get_interpolated_baseline(self, strata_value: str, t: float) -> float:
        if not self._baseline_by_strata:
            return 0.0
        key = str(strata_value)
        if key not in self._baseline_by_strata:
            key = next(iter(self._baseline_by_strata.keys()))
        s = self._baseline_by_strata[key]
        x = s.index.values.astype(float)
        y = s.values.astype(float)
        if len(x) == 0:
            return 0.0
        if t <= x.min():
            return float(y[0])
        if t >= x.max():
            return float(y[-1])
        return float(np.interp(t, x, y))

    def predict_partial_hazard(self, df: pd.DataFrame) -> pd.Series:
        artifacts = self._get_artifacts()
        X_df, _, _ = self._prepare_feature_matrix(
            df,
            artifacts.feature_cols,
            feature_medians=artifacts.feature_medians,
            scaler=artifacts.scaler,
            fit=False,
        )
        hazard = artifacts.model.predict_partial_hazard(X_df)
        return pd.Series(np.asarray(hazard).reshape(-1), index=df.index, name="partial_hazard")

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        artifacts = self._get_artifacts()
        X_df, _, _ = self._prepare_feature_matrix(
            df,
            artifacts.feature_cols,
            feature_medians=artifacts.feature_medians,
            scaler=artifacts.scaler,
            fit=False,
        )
        return X_df

    def linear_predictor(self, df: pd.DataFrame) -> pd.Series:
        artifacts = self._get_artifacts()
        X_df = self.transform_features(df)
        beta = artifacts.model.params_.reindex(artifacts.feature_cols).fillna(0.0)
        lp = X_df.mul(beta, axis=1).sum(axis=1)
        return pd.Series(lp.values, index=df.index, name="linear_predictor")

    def local_feature_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        artifacts = self._get_artifacts()
        X_df = self.transform_features(df)
        beta = artifacts.model.params_.reindex(artifacts.feature_cols).fillna(0.0)
        contrib = X_df.mul(beta, axis=1)
        contrib.index = df.index
        return contrib

    def predict_default_probability(self, df: pd.DataFrame, horizon_days: int = 90) -> pd.Series:
        scores = self.predict_partial_hazard(df)
        probs = []
        for idx, row in df.iterrows():
            t = float(row["stop"])
            strata_val = row[self.strata_col]
            h_t = self._get_interpolated_baseline(strata_val, t)
            h_th = self._get_interpolated_baseline(strata_val, t + horizon_days)
            delta = max(h_th - h_t, 0.0)
            p = 1.0 - np.exp(-delta * float(scores.loc[idx]))
            probs.append(float(np.clip(p, 0.0, 1.0)))
        return pd.Series(probs, index=df.index, name=f"p_default_{horizon_days}d")

    def concordance_on_loans(self, loan_df: pd.DataFrame, score_col: str = "risk_score") -> float:
        if loan_df.empty:
            return float("nan")
        return float(
            concordance_index(
                event_times=loan_df["duration_days"].astype(float),
                predicted_scores=-loan_df[score_col].astype(float),
                event_observed=loan_df["event"].astype(int),
            )
        )
