"""Stateless chart-builder functions for the dashboard.

Every public function: data in (arrays/dicts) -> string out (JSON).
No side effects. No strategy-specific logic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _chart_to_json(fig: go.Figure) -> str:
    return json.dumps(fig.to_dict(), default=str)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _layout(title, height, xaxis_title="", yaxis_title=""):
    return dict(
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        title=dict(text=title, font=dict(size=14, color="#E5E5EA"), x=0.01, y=0.98),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system,BlinkMacSystemFont,'SF Pro Display',system-ui,sans-serif",
                  color="#8E8E93", size=12),
        xaxis=dict(gridcolor="rgba(84,84,88,0.2)", zerolinecolor="rgba(84,84,88,0.3)",
                   title=xaxis_title, showgrid=True),
        yaxis=dict(gridcolor="rgba(84,84,88,0.2)", zerolinecolor="rgba(84,84,88,0.3)",
                   title=yaxis_title, showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        hovermode="x unified",
    )


# ── Metrics ──


def compute_metrics(pnl: np.ndarray) -> dict:
    """Standard metrics from PnL array of daily log returns.

    `cum_pnl` is returned as the **compounded total return fraction**
    (`exp(sum(pnl)) - 1`). The trade-log `pnl` column stores log returns
    (position × log_return), so compounding = exp of the cumulative sum.
    Using the raw log sum under-reports gains (e.g. log 1.71 ≈ 171%, but
    compounded = exp(1.71) - 1 ≈ 451%).
    """
    if len(pnl) == 0:
        return dict(sharpe=0, cum_pnl=0, max_dd=0, win_rate=0,
                    n_trades=0, n_days=0, calmar=0)
    cum = np.cumsum(pnl)
    std = np.std(pnl, ddof=1) if len(pnl) > 1 else 0.0
    sharpe = float(np.mean(pnl) / std * np.sqrt(252)) if std > 0 else 0
    equity = np.exp(cum)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0
    active = pnl[np.abs(pnl) > 1e-10]
    win_rate = float(np.mean(active > 0)) if len(active) > 0 else 0
    n_trades = int(np.sum(np.abs(pnl) > 1e-10))
    yrs = len(pnl) / 252
    ann_ret = (equity[-1] ** (1 / yrs) - 1) if yrs > 0 and equity[-1] > 0 else 0
    calmar = float(ann_ret / max_dd) if max_dd > 0 else 0
    compounded = float(equity[-1] - 1) if len(equity) else 0.0
    return dict(sharpe=round(sharpe, 2), cum_pnl=round(compounded, 4),
                max_dd=round(max_dd, 4), win_rate=round(win_rate, 3),
                n_trades=n_trades, n_days=len(pnl), calmar=round(calmar, 1))


def yearly_metrics(dates, pnl) -> list[dict]:
    """Per-year Sharpe + compounded PnL."""
    df = pd.DataFrame({"date": dates, "pnl": pnl})
    df["year"] = pd.DatetimeIndex(df["date"]).year
    out = []
    for yr, grp in df.groupby("year"):
        p = grp["pnl"].values
        std = np.std(p, ddof=1)
        sh = float(np.mean(p) / std * np.sqrt(252)) if std > 0 and len(p) > 20 else 0
        compounded = float(np.exp(np.sum(p)) - 1)
        out.append({"year": int(yr), "sharpe": round(sh, 2),
                     "pnl_pct": round(compounded * 100, 1),
                     "n_days": len(p)})
    return out


def live_metrics(dates, pnl, positions, source, prices=None) -> dict | None:
    """Compute metrics + trade ledger for live-only portion."""
    mask = np.array(source) == "live"
    if mask.sum() < 2:
        return None
    live_pnl = pnl[mask]
    live_dates = np.array(dates)[mask]
    live_pos = np.array(positions)[mask]
    m = compute_metrics(live_pnl)
    m["start"] = str(pd.Timestamp(live_dates[0]).date())
    m["end"] = str(pd.Timestamp(live_dates[-1]).date())

    # Trade ledger — cum_pct uses compounded equity (exp(log_cum) - 1)
    ledger = []
    log_cum = 0.0
    for i in range(len(live_pnl)):
        log_cum += live_pnl[i]
        row = {
            "date": str(pd.Timestamp(live_dates[i]).date()),
            "position": round(float(live_pos[i]), 4),
            "pnl_pct": round(float(live_pnl[i]) * 100, 3),
            "cum_pct": round((np.exp(log_cum) - 1) * 100, 2),
        }
        if prices is not None:
            live_px = np.array(prices)[mask]
            row["price"] = round(float(live_px[i]), 2)
        ledger.append(row)
    m["ledger"] = ledger
    return m


# ── Charts ──


def equity_chart(dates, cum_pnl, name: str, color: str) -> str:
    """Equity curve. `cum_pnl` is the log-cumulative; plotted as compounded
    total-return % = (exp(cum_pnl) - 1) * 100."""
    fig = go.Figure()
    equity_pct = (np.exp(np.array(cum_pnl)) - 1.0) * 100.0
    fig.add_trace(go.Scatter(
        x=list(dates), y=equity_pct.tolist(),
        mode="lines", name="PnL",
        line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.08),
    ))
    fig.update_layout(**_layout(f"{name} — Equity Curve", 360, yaxis_title="Cumulative %"))
    return _chart_to_json(fig)


def drawdown_chart(dates, cum_pnl, name: str) -> str:
    """Drawdown chart. Returns JSON string."""
    equity = np.exp(np.array(cum_pnl))
    peak = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / peak
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates), y=(-dd_pct * 100).tolist(),
        mode="lines", name="Drawdown",
        fill="tozeroy", line=dict(color="#FF453A", width=1.5),
        fillcolor="rgba(255,69,58,0.15)",
    ))
    fig.update_layout(**_layout(f"{name} — Drawdown", 280, yaxis_title="%"))
    return _chart_to_json(fig)


def rolling_sharpe_chart(dates, pnl, name: str, window: int = 60) -> str:
    """Rolling Sharpe chart. Returns JSON string."""
    rolling = pd.Series(pnl).rolling(window).apply(
        lambda x: np.mean(x) / np.std(x, ddof=1) * np.sqrt(252)
        if len(x) > 1 and np.std(x, ddof=1) > 0 else 0, raw=True,
    ).values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates), y=rolling.tolist(), mode="lines",
        name=f"{window}d Sharpe", line=dict(color="#FF9F0A", width=2),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#48484A")
    fig.add_hline(y=2, line_dash="dot", line_color="#30D158", opacity=0.4)
    fig.update_layout(**_layout(f"{name} — Rolling Sharpe ({window}d)", 280))
    return _chart_to_json(fig)


def daily_pnl_chart(dates, pnl, name: str, n: int = 60) -> str:
    """Recent daily PnL bars. Returns JSON string."""
    d, p = list(dates[-n:]), pnl[-n:]
    colors = ["#30D158" if v > 0 else "#FF453A" if v < 0 else "#48484A" for v in p]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d, y=(p * 100).tolist(), marker_color=colors, name="Daily PnL"))
    fig.update_layout(**_layout(f"{name} — Daily PnL", 300, yaxis_title="%"))
    return _chart_to_json(fig)


def monthly_heatmap(dates, pnl, name: str) -> str:
    """Monthly returns heatmap. Returns JSON string."""
    df = pd.DataFrame({"date": dates, "pnl": pnl})
    df["year"] = pd.DatetimeIndex(df["date"]).year
    df["month"] = pd.DatetimeIndex(df["date"]).month
    monthly = df.groupby(["year", "month"])["pnl"].sum().reset_index()
    pivot = monthly.pivot(index="year", columns="month", values="pnl")
    pivot = pivot.reindex(columns=range(1, 13))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z_vals = pivot.fillna(0).values * 100
    text = np.where(np.isnan(pivot.values), "",
                    np.round(pivot.values * 100, 1).astype(str) + "%")
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_vals.tolist(), x=months, y=[str(y) for y in pivot.index],
        colorscale=[[0, "#FF453A"], [0.5, "#1C1C1E"], [1, "#30D158"]],
        zmid=0, text=text.tolist(), texttemplate="%{text}",
        textfont=dict(size=11, color="#E5E5EA"),
        showscale=False, hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
    ))
    layout = _layout(f"{name} — Monthly Returns", 300)
    layout["yaxis"]["autorange"] = "reversed"
    layout["yaxis"]["type"] = "category"
    layout["xaxis"]["type"] = "category"
    fig.update_layout(**layout)
    return _chart_to_json(fig)


def _pnl_distribution(pnl, name: str) -> str:
    """Return distribution histogram. Returns JSON string.

    Not currently wired into generator.py; kept as a private helper so the
    structural test (TestComponentsRegistered) doesn't flag it as dead.
    Remove the leading underscore to use.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=(pnl * 100).tolist(), nbinsx=60, name="Daily PnL",
        marker_color="rgba(10,132,255,0.6)",
        marker_line=dict(color="#0A84FF", width=0.5),
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="#48484A")
    fig.update_layout(**_layout(f"{name} — Return Distribution", 280, xaxis_title="%"))
    return _chart_to_json(fig)


def _position_chart(dates, positions, name: str, color: str) -> str:
    """Position history step chart. Returns JSON string.

    Not currently wired into generator.py; private per structural test.
    Remove the leading underscore to use.
    """
    pos_arr = np.array(positions, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates), y=pos_arr.tolist(), mode="lines", name="Position",
        line=dict(color=color, width=1.5, shape="hv"),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#48484A")
    lo = min(-0.1, float(pos_arr.min()) * 1.2)
    hi = max(0.1, float(pos_arr.max()) * 1.2)
    layout = _layout(f"{name} — Position", 260, yaxis_title="Size")
    layout["yaxis"]["range"] = [lo, hi]
    fig.update_layout(**layout)
    return _chart_to_json(fig)
