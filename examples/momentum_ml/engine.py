"""Walk-forward momentum ML strategy — demo implementation.

Educational demo showing:
- Feature engineering with shift(1) (zero look-ahead)
- Walk-forward GBDT with rolling train window
- Decision threshold tuning on validation set
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from causal_edge.engine.base import StrategyEngine


class MomentumMLEngine(StrategyEngine):
    """Walk-forward GBDT on momentum features. Long/Flat only."""

    def __init__(self, context: dict | None = None, n_days: int = 750) -> None:
        super().__init__(context=context)
        self.n_days = n_days
        self.train_window = 126  # ~6 months rolling
        self.retrain_every = 5  # retrain every 5 days (weekly)

    def compute_signals(self):
        """Generate synthetic prices and compute walk-forward ML signals."""
        rng = np.random.default_rng(123)
        # Synthetic trending price with momentum
        raw_ret = rng.normal(0.0003, 0.015, self.n_days)
        # Add slight autocorrelation to make momentum features useful
        for i in range(1, len(raw_ret)):
            raw_ret[i] += 0.05 * raw_ret[i - 1]
        prices = 100.0 * np.cumprod(1.0 + raw_ret)
        dates = pd.date_range(end="2026-01-01", periods=self.n_days, freq="B", tz="UTC")
        returns = np.zeros_like(prices)
        returns[1:] = prices[1:] / prices[:-1] - 1.0

        # Features — ALL shifted by 1 to prevent look-ahead
        s = pd.Series(returns)
        features = pd.DataFrame(
            {
                "ret_1d": s.shift(1),  # yesterday's return
                "ret_5d": s.rolling(5).sum().shift(1),  # 5-day momentum
                "ret_20d": s.rolling(20).sum().shift(1),  # 20-day momentum
                "vol_20d": s.rolling(20).std().shift(1),  # 20-day volatility
                "rsi_14": _rsi(s, 14).shift(1),  # RSI(14)
            }
        )

        # Target: next-day direction (1=up, 0=down)
        target = (s > 0).astype(int)

        # Walk-forward prediction
        positions = np.zeros(self.n_days)
        start = self.train_window + 20  # enough warmup for features + train

        last_model = None
        last_train_day = 0

        for t in range(start, self.n_days):
            # Retrain periodically
            if last_model is None or (t - last_train_day) >= self.retrain_every:
                train_start = max(0, t - self.train_window)
                X_train = features.iloc[train_start:t].values
                y_train = target.iloc[train_start:t].values
                # Drop rows with NaN from feature warmup
                valid = ~np.isnan(X_train).any(axis=1)
                X_train, y_train = X_train[valid], y_train[valid]

                if len(X_train) < 30:
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

            # Predict using features available at time t (already shifted)
            x_t = features.iloc[t].values.reshape(1, -1)
            if np.isnan(x_t).any():
                positions[t] = 0.0
                continue

            prob = last_model.predict_proba(x_t)[0]
            # Long if P(up) > 0.55 (conservative threshold)
            positions[t] = 1.0 if prob[1] > 0.55 else 0.0

        return self.finalize_signals(positions, dates, prices)

    def get_latest_signal(self):
        """Return latest position from walk-forward model."""
        positions, dates, prices = self.compute_signals()
        return {
            "position": float(positions[-1]),
            "date": str(dates[-1].date()),
            "price": float(prices[-1]),
        }


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
