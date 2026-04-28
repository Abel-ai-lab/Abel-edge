"""DecisionContext walk-forward momentum ML example."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from abel_edge.engine.base import StrategyEngine


class MomentumMLEngine(StrategyEngine):
    """Walk-forward GBDT on target-close features. Long/Flat only."""

    def __init__(self, context: dict | None = None) -> None:
        super().__init__(context=context)
        self.train_window = 126  # ~6 months rolling
        self.retrain_every = 5  # retrain every 5 days (weekly)

    def compute_decisions(self, ctx):
        close = ctx.target.series("close").astype(float)
        returns = close.pct_change().fillna(0.0)
        features = pd.DataFrame(
            {
                "ret_1d": returns,
                "ret_5d": returns.rolling(5, min_periods=5).sum(),
                "ret_20d": returns.rolling(20, min_periods=20).sum(),
                "vol_20d": returns.rolling(20, min_periods=20).std(),
                "sma_gap_10": close / close.rolling(10, min_periods=10).mean() - 1.0,
                "rsi_14": _rsi(returns, 14),
            },
            index=close.index,
        )
        target = (returns.shift(-1) > 0).astype(int)

        next_position = pd.Series(0.0, index=close.index, dtype=float)
        start = max(self.train_window, 25)

        last_model = None
        last_train_day = 0

        for t in range(start, len(close)):
            if last_model is None or (t - last_train_day) >= self.retrain_every:
                train_start = max(0, t - self.train_window)
                train_slice = features.iloc[train_start:t]
                target_slice = target.iloc[train_start:t]
                valid = (~train_slice.isna().any(axis=1)) & target_slice.notna()
                if int(valid.sum()) < 30:
                    continue
                X_train = train_slice.loc[valid].to_numpy()
                y_train = target_slice.loc[valid].to_numpy()

                if len(np.unique(y_train)) < 2:
                    continue

                model = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                last_model = model
                last_train_day = t

            x_t = features.iloc[t].to_numpy(dtype=float).reshape(1, -1)
            if np.isnan(x_t).any():
                continue

            prob = last_model.predict_proba(x_t)[0]
            next_position.iloc[t] = 1.0 if prob[1] > 0.55 else 0.0

        return ctx.decisions(next_position)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
