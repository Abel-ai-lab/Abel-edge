from __future__ import annotations

import pandas as pd


def filter_tracking_rows(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0 or "source" not in df.columns:
        return None
    tracked = df[df["source"].astype(str).str.lower() == "live"].copy()
    return tracked if len(tracked) > 0 else None


def filter_backtest_rows(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    if "source" not in df.columns:
        return df.copy()
    backtest = df[df["source"].astype(str).str.lower() != "live"].copy()
    return backtest if len(backtest) > 0 else None


def paper_rows(df: pd.DataFrame | None, *, signal_label, fmt_pnl_pct) -> list[dict]:
    tracked = filter_tracking_rows(df)
    if tracked is None:
        return []
    rows = tracked.sort_values("date", ascending=False).head(8)
    result = []
    for _, row in rows.iterrows():
        result.append(
            {
                "date": pd.to_datetime(row["date"]).date().isoformat(),
                "close": "N/A" if pd.isna(row.get("close")) else f"{float(row['close']):.2f}",
                "signal": signal_label(float(row.get("next_position", row.get("position", 0.0)))),
                "next_position": f"{float(row.get('next_position', row.get('position', 0.0))):.2f}",
                "pnl": fmt_pnl_pct(float(row.get("pnl", 0.0))),
            }
        )
    return result
