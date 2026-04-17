"""Abstract base class for strategy engines.

Every strategy engine must implement compute_signals() and get_latest_signal().
Engines are standalone: strategies/ never imports causal_edge/ internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from causal_edge.engine.feed_contract import align_series_to_dates
from causal_edge.engine.feed_loader import load_declared_feed
from causal_edge.engine.signal_contract import validate_signal_output


class StrategyEngine(ABC):
    """Base class for all strategy engines.

    Subclasses must implement:
        compute_signals() -> tuple of (positions, dates, prices)
        get_latest_signal() -> dict with at least 'position' key
    """

    def __init__(self, context: dict | None = None) -> None:
        self.context = context

    def research_context(self) -> dict:
        """Return the research-specific context payload injected by the evaluator."""
        research = (self.context or {}).get("_research")
        return dict(research) if isinstance(research, dict) else {}

    def research_requested_window(self) -> dict:
        """Return the requested evaluation window for this research run."""
        requested = self.research_context().get("requested_window")
        return dict(requested) if isinstance(requested, dict) else {"start": None, "end": None}

    def research_requested_start(self) -> str | None:
        """Return the requested research start date when present."""
        return self.research_requested_window().get("start")

    def research_discovery(self) -> dict:
        """Return the injected discovery payload for this branch, if any."""
        discovery = (self.context or {}).get("discovery")
        return dict(discovery) if isinstance(discovery, dict) else {}

    def research_data_readiness(self) -> dict:
        """Return the injected edge-owned data-readiness report for discovery tickers."""
        readiness = self.research_discovery().get("data_readiness")
        return dict(readiness) if isinstance(readiness, dict) else {}

    def research_target_ticker(self) -> str | None:
        """Return the normalized research target ticker."""
        discovery = self.research_discovery()
        ticker = discovery.get("ticker") or (self.context or {}).get("ticker")
        if ticker is None:
            return None
        return str(ticker).strip().upper() or None

    def research_driver_candidates(
        self,
        *,
        roles: tuple[str, ...] | list[str] | None = None,
        require_usable: bool = True,
        require_full_window: bool = False,
        exclude_target: bool = True,
    ) -> list[dict]:
        """Return discovered driver candidates merged with edge data-readiness facts."""
        discovery = self.research_discovery()
        target = self.research_target_ticker()
        readiness_by_ticker = _readiness_by_ticker(self.research_data_readiness())
        allowed_roles = {str(role).strip().lower() for role in (roles or []) if str(role).strip()}
        candidates = _discovery_candidates(discovery)

        merged: list[dict] = []
        for item in candidates:
            ticker = item["ticker"]
            if exclude_target and target and ticker == target:
                continue
            discovery_roles = item["discovery_roles"]
            if allowed_roles and not allowed_roles.intersection({role.lower() for role in discovery_roles}):
                continue

            readiness = readiness_by_ticker.get(ticker, {})
            usable = bool(readiness.get("usable", False))
            full_window = bool(readiness.get("full_window", False))
            if require_usable and not usable:
                continue
            if require_full_window and not full_window:
                continue

            merged.append(
                {
                    "ticker": ticker,
                    "field": item.get("field"),
                    "discovery_roles": discovery_roles,
                    "readiness_status": readiness.get("status", "unknown"),
                    "usable": usable,
                    "full_window": full_window,
                    "rows": int(readiness.get("rows", 0) or 0),
                    "first_timestamp": readiness.get("first_timestamp"),
                    "last_timestamp": readiness.get("last_timestamp"),
                    "note": readiness.get("note"),
                }
            )
        return merged

    def research_driver_tickers(self, **kwargs) -> list[str]:
        """Return only the tickers for research driver candidates."""
        return [item["ticker"] for item in self.research_driver_candidates(**kwargs)]

    def load_research_bars(
        self,
        *,
        driver_tickers: list[str] | None = None,
        include_target: bool = True,
        require_usable: bool = True,
        require_full_window: bool = False,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = 600,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load a research-ready bar set for the target plus selected drivers."""
        symbols = self._research_symbol_list(
            driver_tickers=driver_tickers,
            include_target=include_target,
            require_usable=require_usable,
            require_full_window=require_full_window,
        )
        return self.load_bars(
            symbols=symbols or None,
            start=self.research_requested_start() if start is None else start,
            end=end,
            timeframe=timeframe,
            limit=limit,
            fields=fields,
        )

    def research_close_frame(
        self,
        *,
        driver_tickers: list[str] | None = None,
        include_target: bool = True,
        require_usable: bool = True,
        require_full_window: bool = False,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = 600,
    ) -> pd.DataFrame:
        """Load a close-price frame ordered by target then driver ticker selection."""
        symbols = self._research_symbol_list(
            driver_tickers=driver_tickers,
            include_target=include_target,
            require_usable=require_usable,
            require_full_window=require_full_window,
        )
        bars = self.load_research_bars(
            driver_tickers=driver_tickers,
            include_target=include_target,
            require_usable=require_usable,
            require_full_window=require_full_window,
            start=start,
            end=end,
            timeframe=timeframe,
            limit=limit,
            fields=["close"],
        )
        if bars.empty:
            raise ValueError("No bars returned for the selected research target/driver set.")
        if "close" not in bars.columns:
            raise ValueError("Research bars did not include a 'close' column.")
        frame = (
            bars.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
            .sort_index()
        )
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        ordered_columns = [symbol for symbol in symbols if symbol in frame.columns]
        frame = frame.loc[:, ordered_columns]
        target = self.research_target_ticker()
        if include_target and target and target not in frame.columns:
            raise ValueError(
                f"Research close frame is missing the target ticker '{target}'. "
                "Do not drop or filter the target out before validating the frame."
            )
        return frame

    def research_target_driver_frame(
        self,
        *,
        driver_tickers: list[str] | None = None,
        require_usable: bool = True,
        require_full_window: bool = False,
        overlap: str = "intersection",
        require_drivers: bool = False,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = 600,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Return a safely prepared target series plus aligned driver frame.

        overlap:
            - "intersection": keep only timestamps where target and every selected
              driver are present.
            - "target_only": keep timestamps where the target is present and leave
              driver gaps explicit for the engine to handle intentionally.
        """
        target = self.research_target_ticker()
        if not target:
            raise ValueError("Research target ticker is not available in the injected context.")
        if overlap not in {"intersection", "target_only"}:
            raise ValueError(
                f"Unsupported overlap mode '{overlap}'. Supported: 'intersection', 'target_only'."
            )

        frame = self.research_close_frame(
            driver_tickers=driver_tickers,
            include_target=True,
            require_usable=require_usable,
            require_full_window=require_full_window,
            start=start,
            end=end,
            timeframe=timeframe,
            limit=limit,
        )
        if target not in frame.columns:
            raise ValueError(f"Research close frame is missing the target ticker '{target}'.")

        aligned = frame.dropna(subset=[target]).copy()
        driver_frame = aligned.drop(columns=[target], errors="ignore")
        if require_drivers and driver_frame.shape[1] == 0:
            raise ValueError("No driver columns remain after applying the research driver filters.")
        if overlap == "intersection" and driver_frame.shape[1] > 0:
            aligned = aligned.dropna()
            driver_frame = aligned.drop(columns=[target], errors="ignore")
        if aligned.empty:
            if overlap == "intersection":
                raise ValueError(
                    "No overlapping target/driver rows survived the selected research frame. "
                    "Trim the driver list or relax the overlap mode before continuing."
                )
            raise ValueError(
                "No target rows survived the selected research frame. "
                "Check the requested window and target data availability."
            )

        target_series = aligned[target].astype(float)
        driver_frame = driver_frame.astype(float)
        target_series.index = pd.DatetimeIndex(pd.to_datetime(target_series.index, utc=True))
        driver_frame.index = pd.DatetimeIndex(pd.to_datetime(driver_frame.index, utc=True))
        return target_series, driver_frame

    def bind_price_loader(self, loader, price_data_config: dict | None = None) -> None:
        """Deprecated legacy loader hook.

        Primary bars now come from the synthesized framework-managed `primary`
        feed, and all external data must flow through `price_data` / `feeds`
        plus `load_bars()` / `load_feed()`.
        """
        raise RuntimeError(
            "StrategyEngine.bind_price_loader() is deprecated and no longer supported. "
            "Declare primary data via strategy price_data, declare auxiliary inputs via feeds, "
            "and load them with load_bars() / load_feed()."
        )

    def load_bars(
        self,
        symbols: list[str] | None = None,
        *,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        bars = self.load_feed(
            "primary",
            start=start,
            end=end,
            timeframe=timeframe,
            limit=limit,
            fields=fields,
        )
        if symbols is None:
            return bars
        return bars[bars["symbol"].isin(symbols)].reset_index(drop=True)

    def load_feed(
        self,
        name: str,
        *,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        return load_declared_feed(
            self,
            name,
            start=start,
            end=end,
            timeframe=timeframe,
            limit=limit,
            fields=fields,
        )

    def feed_series(
        self,
        name: str,
        field: str = "close",
        *,
        align_to=None,
        method: str | None = None,
        allow_gaps: bool = True,
    ) -> pd.Series:
        frame = self.load_feed(name)
        feed_cfg = ((self.context or {}).get("_feeds") or {}).get(name) or {}
        kind = feed_cfg.get("kind")
        if kind == "series":
            series = pd.Series(frame["value"].to_numpy(), index=pd.DatetimeIndex(frame["timestamp"]))
        else:
            if field not in frame.columns:
                raise ValueError(f"Feed '{name}' does not expose field '{field}'.")
            series = pd.Series(frame[field].to_numpy(), index=pd.DatetimeIndex(frame["timestamp"]))
        if align_to is not None:
            series = self.align_series(
                series,
                align_to,
                method="ffill" if method is None else method,
                allow_gaps=allow_gaps,
            )
        return series

    def align_series(
        self,
        series: pd.Series,
        dates,
        *,
        method: str | None = "ffill",
        allow_gaps: bool = True,
    ) -> pd.Series:
        profile = ((self.context or {}).get("_data_contract") or {}).get("profile", "daily")
        return align_series_to_dates(
            series,
            dates,
            profile=profile,
            method=method,
            allow_gaps=allow_gaps,
        )

    def finalize_signals(
        self,
        positions,
        dates,
        prices,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        profile = ((self.context or {}).get("_data_contract") or {}).get("profile", "daily")
        return validate_signal_output(
            positions,
            dates,
            prices,
            profile=profile,
        )

    @abstractmethod
    def compute_signals(
        self,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        """Compute full signal history.

        Returns:
            Tuple of (positions, dates, prices) where:
                positions: np.ndarray of daily position sizes (0=flat, 1=long).
                    IMPORTANT: positions[t] must be decided using only data through
                    day t-1. Apply .shift(1) to any indicators before using them
                    to determine positions. This prevents look-ahead bias.
                dates: pd.DatetimeIndex of trading dates
                prices: np.ndarray of daily closing prices
        """

    @abstractmethod
    def get_latest_signal(self) -> dict:
        """Return the most recent signal as a dict.

        Must include at least a 'position' key.
        """

    def get_paper_signal(self, *, as_of=None) -> dict:
        """Return the next position decided at the close of ``as_of``.

        Engines can override this to support incrementally appending live paper-trading rows.
        The returned dict must include at least ``next_position``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement paper trading. "
            "Add get_paper_signal(as_of=...) to the engine."
        )

    def _research_symbol_list(
        self,
        *,
        driver_tickers: list[str] | None,
        include_target: bool,
        require_usable: bool,
        require_full_window: bool,
    ) -> list[str]:
        symbols: list[str] = []
        if include_target:
            target = self.research_target_ticker()
            if target:
                symbols.append(target)
        selected = driver_tickers
        if selected is None:
            selected = self.research_driver_tickers(
                require_usable=require_usable,
                require_full_window=require_full_window,
            )
        for ticker in selected:
            normalized = _normalize_ticker(ticker)
            if normalized and normalized not in symbols:
                symbols.append(normalized)
        return symbols


def _normalize_ticker(value: object) -> str | None:
    ticker = str(value or "").strip().upper()
    return ticker or None


def _discovery_candidates(discovery: dict) -> list[dict]:
    combined: dict[str, dict] = {}

    def remember(item: object, fallback_role: str) -> None:
        if isinstance(item, str):
            ticker = _normalize_ticker(item)
            field = None
            roles = [fallback_role]
        elif isinstance(item, dict):
            ticker = _normalize_ticker(item.get("ticker"))
            field_value = str(item.get("field", "")).strip()
            field = field_value or None
            roles = [str(role).strip() for role in item.get("roles", []) if str(role).strip()]
            if fallback_role not in roles:
                roles.append(fallback_role)
        else:
            return
        if not ticker:
            return
        record = combined.setdefault(
            ticker,
            {"ticker": ticker, "field": field, "discovery_roles": set()},
        )
        if record["field"] is None and field is not None:
            record["field"] = field
        record["discovery_roles"].update(role for role in roles if role)

    for item in discovery.get("parents") or []:
        remember(item, "parent")
    for item in discovery.get("blanket_new") or []:
        remember(item, "blanket")
    for item in discovery.get("children") or []:
        remember(item, "child")

    return [
        {
            "ticker": ticker,
            "field": payload["field"],
            "discovery_roles": sorted(payload["discovery_roles"]),
        }
        for ticker, payload in sorted(combined.items())
    ]


def _readiness_by_ticker(report: dict) -> dict[str, dict]:
    results = report.get("results") or []
    merged: dict[str, dict] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        ticker = _normalize_ticker(item.get("ticker"))
        if not ticker:
            continue
        merged[ticker] = dict(item)
    return merged
