"""Base class for strategy engines.

The branch-default contract is ``compute_decisions(self, ctx)``. Legacy
``compute_signals()`` engines are still supported internally during the rollout
so existing examples and tests remain runnable.
"""

from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd

from abel_edge.engine.decision_context import DecisionContext
from abel_edge.engine.feed_contract import align_series_to_dates
from abel_edge.engine.feed_loader import load_declared_feed, load_feed_frame
from abel_edge.graph_nodes import coerce_graph_node_refs
from abel_edge.engine.runtime_contract import (
    CompiledDecisionOutput,
    DecisionDraft,
    ExecutionConstraints,
    RuntimeProfile,
    compile_decision_draft,
    execution_constraints_from_context,
    legacy_output_to_compiled,
    runtime_profile_from_context,
)
from abel_edge.engine.signal_contract import validate_signal_output


def _coerce_bootstrap_cutover(value) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        raise ValueError("paper bootstrap cutover_as_of is required")
    return pd.to_datetime(value, utc=True)


class StrategyEngine(ABC):
    """Base class for all strategy engines.

    Preferred subclass contract:
        compute_decisions(ctx) -> DecisionDraft

    Legacy contract:
        compute_signals() -> tuple of (positions, dates, prices)
    """

    def __init__(self, context: dict | None = None) -> None:
        self.context = context
        self._decision_surface_guard = False
        self._last_decision_context: DecisionContext | None = None
        self._last_compiled_output: CompiledDecisionOutput | None = None
        self._paper_bootstrap_cutover_as_of = None

    def runtime_profile(self) -> RuntimeProfile:
        """Return the explicit runtime profile for this run."""
        return runtime_profile_from_context(self.context)

    def execution_constraints(self) -> ExecutionConstraints:
        """Return the execution envelope for this run."""
        return execution_constraints_from_context(self.context)

    def prepared_input_nodes(self) -> list[str]:
        """Return prepared non-primary input feed names when present."""
        feeds = (self.context or {}).get("_feeds") or {}
        return sorted(
            str(name)
            for name in feeds.keys()
            if str(name).strip() and str(name) != "primary"
        )

    def prepared_input_specs(self) -> list[dict]:
        """Return prepared non-primary feed specs for authoring/debugging helpers."""
        feeds = (self.context or {}).get("_feeds") or {}
        specs: list[dict] = []
        for name in self.prepared_input_nodes():
            cfg = dict(feeds.get(name) or {})
            cfg.setdefault("name", name)
            specs.append(cfg)
        return specs

    def decision_context(
        self,
        *,
        start=None,
        end=None,
        limit: int | None = None,
    ) -> DecisionContext:
        """Construct the branch-visible decision context."""
        return self._decision_context(
            start=start,
            end=end,
            limit=limit,
            apply_paper_window=True,
        )

    def paper_bootstrap_context(
        self,
        *,
        start=None,
        end=None,
        limit: int | None = None,
    ) -> DecisionContext:
        """Construct a context for hosted paper startup-state bootstrap reads.

        Daily paper execution may apply ``paperExecutionProfile.history`` to
        bound market-data reads. Startup-state bootstrap can need a different
        range to reconstruct a correct cutover state, so this helper keeps the
        same runtime feeds and paths while bypassing the daily paper window.
        """
        return self._decision_context(
            start=start,
            end=end,
            limit=limit,
            apply_paper_window=False,
        )

    @contextmanager
    def paper_bootstrap_cutover_scope(self, cutover_as_of) -> Iterator[None]:
        """Bound bootstrap-time data contexts to a validation cutover."""
        cutover = _coerce_bootstrap_cutover(cutover_as_of)
        previous = self._paper_bootstrap_cutover_as_of
        self._paper_bootstrap_cutover_as_of = cutover
        try:
            yield
        finally:
            self._paper_bootstrap_cutover_as_of = previous

    def _decision_context(
        self,
        *,
        start=None,
        end=None,
        limit: int | None = None,
        apply_paper_window: bool,
    ) -> DecisionContext:
        end = self._bootstrap_bounded_end(end)
        return DecisionContext(
            self,
            runtime_profile=self.runtime_profile(),
            execution_constraints=self.execution_constraints(),
            start=self.research_requested_start() if start is None else start,
            end=end,
            limit=limit,
            apply_paper_window=apply_paper_window,
        )

    def _bootstrap_bounded_end(self, end):
        cutover = self._paper_bootstrap_cutover_as_of
        if cutover is None:
            return end
        if end is None:
            return cutover
        end_ts = _coerce_bootstrap_cutover(end)
        cutover_ts = pd.to_datetime(cutover, utc=True)
        if end_ts > cutover_ts:
            raise ValueError(
                "paper bootstrap context end "
                f"{end_ts.date().isoformat()} is after validation cutover "
                f"{cutover_ts.date().isoformat()}"
            )
        return end

    def uses_decision_contract(self) -> bool:
        """Return whether the subclass overrides ``compute_decisions``."""
        return type(self).compute_decisions is not StrategyEngine.compute_decisions

    def uses_legacy_signal_contract(self) -> bool:
        """Return whether the subclass overrides ``compute_signals``."""
        return type(self).compute_signals is not StrategyEngine.compute_signals

    def compute_runtime_output(
        self,
        *,
        start=None,
        end=None,
        limit: int | None = None,
    ) -> CompiledDecisionOutput:
        """Run the active contract and normalize into one compiled output shape."""
        if self.uses_decision_contract():
            ctx = self.decision_context(start=start, end=end, limit=limit)
            self._last_decision_context = ctx
            self._decision_surface_guard = True
            try:
                draft = self.compute_decisions(ctx)
            finally:
                self._decision_surface_guard = False
            if not isinstance(draft, DecisionDraft):
                raise TypeError(
                    f"{self.__class__.__name__}.compute_decisions(ctx) must return "
                    "ctx.decisions(...), not a raw array/object."
                )
            close = ctx.target.series("close").to_numpy(dtype=float)
            compiled = compile_decision_draft(
                draft,
                close,
                runtime_profile=ctx.runtime_profile,
                execution_constraints=ctx.execution_constraints,
            )
            self._last_compiled_output = compiled
            return compiled

        if self.uses_legacy_signal_contract():
            profile = self.runtime_profile()
            positions, dates, prices = validate_signal_output(
                *self.compute_signals(),
                profile=profile.profile,
            )
            compiled = legacy_output_to_compiled(
                positions,
                dates,
                prices,
                runtime_profile=profile,
                execution_constraints=self.execution_constraints(),
            )
            self._last_compiled_output = compiled
            return compiled

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either compute_decisions(ctx) "
            "or compute_signals()."
        )

    def research_context(self) -> dict:
        """Return the research-specific context payload injected by the evaluator."""
        research = (self.context or {}).get("_research")
        return dict(research) if isinstance(research, dict) else {}

    def research_branch_spec(self) -> dict:
        """Return the explicit branch specification injected by Abel-alpha."""
        spec = (self.context or {}).get("branch_spec")
        return dict(spec) if isinstance(spec, dict) else {}

    def research_dependencies(self) -> dict:
        """Return the prepared branch dependency payload when available."""
        payload = (self.context or {}).get("dependencies")
        return dict(payload) if isinstance(payload, dict) else {}

    def research_requested_window(self) -> dict:
        """Return the requested evaluation window for this research run."""
        requested = self.research_context().get("requested_window")
        window = dict(requested) if isinstance(requested, dict) else {"start": None, "end": None}
        branch_start = self.research_branch_spec().get("requested_start")
        if branch_start:
            window["start"] = str(branch_start)
        return window

    def research_requested_start(self) -> str | None:
        """Return the requested research start date when present."""
        return self.research_requested_window().get("start")

    def research_discovery(self) -> dict:
        """Return the injected discovery payload for this branch, if any."""
        discovery = (self.context or {}).get("discovery")
        return dict(discovery) if isinstance(discovery, dict) else {}

    def research_data_readiness(self) -> dict:
        """Return the injected edge-owned data-readiness report for discovery tickers."""
        readiness = (self.context or {}).get("readiness")
        if not isinstance(readiness, dict):
            readiness = self.research_discovery().get("data_readiness")
        return dict(readiness) if isinstance(readiness, dict) else {}

    def research_target_ticker(self) -> str | None:
        """Return the normalized research target ticker."""
        branch_spec = self.research_branch_spec()
        ticker = branch_spec.get("target")
        if ticker is not None:
            normalized = str(ticker).strip().upper()
            if normalized:
                return normalized
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
        target_node = _research_target_node(self.context)
        readiness_by_ticker = _readiness_by_ticker(self.research_data_readiness())
        allowed_roles = {str(role).strip().lower() for role in (roles or []) if str(role).strip()}
        candidates = _discovery_candidates(discovery)
        explicit = _explicit_candidates(self.research_branch_spec())
        explicit_selected = bool(explicit)
        if explicit:
            candidates = explicit

        merged: list[dict] = []
        for item in candidates:
            ticker = item["ticker"]
            node_id = str(item.get("node_id") or "")
            if exclude_target and target_node and node_id == target_node:
                continue
            if exclude_target and not node_id and target and ticker == target:
                continue
            discovery_roles = item["discovery_roles"]
            if allowed_roles and not allowed_roles.intersection({role.lower() for role in discovery_roles}):
                continue

            readiness = readiness_by_ticker.get(ticker, {})
            usable = bool(readiness.get("usable", False))
            covers_requested_start = bool(readiness.get("covers_requested_start", False))
            if not explicit_selected:
                if require_usable and not usable:
                    continue
                if require_full_window and not covers_requested_start:
                    continue

            merged.append(
                {
                    "ticker": ticker,
                    "node_id": node_id,
                    "field": item.get("field"),
                    "discovery_roles": discovery_roles,
                    "readiness_status": readiness.get("status", "unknown"),
                    "usable": usable,
                    "covers_requested_start": covers_requested_start,
                    "rows": int(readiness.get("rows", 0) or 0),
                    "observed_first_timestamp": readiness.get("observed_first_timestamp"),
                    "observed_last_timestamp": readiness.get("observed_last_timestamp"),
                    "note": readiness.get("note"),
                }
            )
        return merged

    def research_driver_tickers(self, **kwargs) -> list[str]:
        """Return only the tickers for research driver candidates."""
        ordered: list[str] = []
        for item in self.research_driver_candidates(**kwargs):
            ticker = item["ticker"]
            if ticker not in ordered:
                ordered.append(ticker)
        return ordered

    def research_data_requirements(self) -> dict:
        """Return the prepared branch data requirements when present."""
        dependencies = self.research_dependencies()
        requirements = dependencies.get("data_requirements")
        if isinstance(requirements, dict):
            return dict(requirements)
        branch_spec = self.research_branch_spec()
        requirements = branch_spec.get("data_requirements")
        return dict(requirements) if isinstance(requirements, dict) else {}

    def _research_feed_defaults(self) -> tuple[str, str, str, dict[str, object]]:
        """Resolve adapter/timeframe/profile/cache defaults for research bars."""
        dependencies = self.research_dependencies()
        cache = dependencies.get("cache")
        cache_payload = dict(cache) if isinstance(cache, dict) else {}
        primary = (((self.context or {}).get("_feeds") or {}).get("primary")) or {}
        requirements = self.research_data_requirements()

        adapter = str(
            cache_payload.get("adapter")
            or primary.get("adapter")
            or "abel"
        ).strip().lower()
        timeframe = str(
            cache_payload.get("timeframe")
            or requirements.get("timeframe")
            or primary.get("timeframe")
            or "1d"
        ).strip().lower()
        profile = str(
            cache_payload.get("profile")
            or primary.get("profile")
            or "daily"
        ).strip().lower()
        options: dict[str, object] = {}
        cache_root = cache_payload.get("cache_root")
        if cache_root:
            options["cache_root"] = cache_root
        return adapter, timeframe, profile, options

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
        self._assert_raw_strategy_access_allowed("load_research_bars()")
        symbols = self._research_symbol_list(
            driver_tickers=driver_tickers,
            include_target=include_target,
            require_usable=require_usable,
            require_full_window=require_full_window,
        )
        if not symbols:
            raise ValueError(
                "No research symbols were selected. Include the target or choose at least one driver."
            )
        adapter, default_timeframe, profile, options = self._research_feed_defaults()
        effective_start = self.research_requested_start() if start is None else start
        effective_timeframe = timeframe or default_timeframe
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            frame = self._load_research_symbol_bars(
                symbol,
                adapter=adapter,
                timeframe=effective_timeframe,
                profile=profile,
                options=options,
                start=effective_start,
                end=end,
                limit=limit,
                fields=fields,
            )
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "symbol", *(fields or [])])
        bars = pd.concat(frames, ignore_index=True)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        bars = bars.dropna(subset=["timestamp"]).copy()
        if "symbol" in bars.columns:
            bars["symbol"] = pd.Categorical(bars["symbol"], categories=symbols, ordered=True)
        bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        if "symbol" in bars.columns:
            bars["symbol"] = bars["symbol"].astype(str)
        return bars.reset_index(drop=True)

    def _load_research_symbol_bars(
        self,
        symbol: str,
        *,
        adapter: str,
        timeframe: str,
        profile: str,
        options: dict[str, object],
        start=None,
        end=None,
        limit: int | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        feed_cfg = {
            "name": f"research:{symbol}",
            "kind": "bars",
            "adapter": adapter,
            "symbol": symbol,
            "timeframe": timeframe,
            "profile": profile,
            **options,
        }
        return load_feed_frame(
            feed_cfg,
            strategy_id=(self.context or {}).get("id"),
            start=start,
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
        self._assert_raw_strategy_access_allowed("research_close_frame()")
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
        self._assert_raw_strategy_access_allowed("research_target_driver_frame()")
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

    def _runtime_load_bars(
        self,
        symbols: list[str] | None = None,
        *,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = None,
        fields: list[str] | None = None,
        apply_paper_window: bool = True,
    ) -> pd.DataFrame:
        bars = self._runtime_load_feed(
            "primary",
            start=start,
            end=end,
            timeframe=timeframe,
            limit=limit,
            fields=fields,
            apply_paper_window=apply_paper_window,
        )
        if symbols is None:
            return bars
        return bars[bars["symbol"].isin(symbols)].reset_index(drop=True)

    def _runtime_load_feed(
        self,
        name: str,
        *,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = None,
        fields: list[str] | None = None,
        apply_paper_window: bool = True,
    ) -> pd.DataFrame:
        request_end = None
        request_limit = None
        if apply_paper_window:
            start, limit = self._apply_paper_data_window(start=start, limit=limit)
            request_end = self._paper_data_request_end(end=end)
            request_limit = self._paper_data_request_limit(
                limit=limit,
                request_end=request_end,
            )
        return load_declared_feed(
            self,
            name,
            start=start,
            end=end,
            request_end=request_end,
            request_limit=request_limit,
            timeframe=timeframe,
            limit=limit,
            fields=fields,
        )

    def _apply_paper_data_window(self, *, start=None, limit: int | None = None):
        window = (self.context or {}).get("_paper_data_window")
        if not isinstance(window, dict):
            return start, limit
        window_start = window.get("start")
        window_limit = window.get("limit")
        if window_start is not None:
            if start is None:
                start = window_start
            else:
                current = pd.to_datetime(start, utc=True)
                boundary = pd.to_datetime(window_start, utc=True)
                start = boundary if boundary > current else start
        if window_limit is not None:
            window_limit = int(window_limit)
            limit = window_limit if limit is None else min(int(limit), window_limit)
        return start, limit

    def _paper_data_request_end(self, *, end=None):
        """Return an adapter/cache horizon while preserving the caller-visible end."""
        if end is None:
            return None
        window = (self.context or {}).get("_paper_data_window")
        if not isinstance(window, dict):
            return None
        cache_end = window.get("cache_end")
        if cache_end is None:
            return None
        try:
            visible_end = pd.to_datetime(end, utc=True)
            horizon = pd.to_datetime(cache_end, utc=True)
        except (TypeError, ValueError):
            return cache_end
        return cache_end if horizon > visible_end else None

    def _paper_data_request_limit(self, *, limit: int | None, request_end=None) -> int | None:
        if limit is None or request_end is None:
            return None
        window = (self.context or {}).get("_paper_data_window")
        if not isinstance(window, dict):
            return None
        try:
            extra_bars = int(window.get("cache_extra_bars") or 0)
        except (TypeError, ValueError):
            extra_bars = 0
        if extra_bars <= 0:
            return None
        return int(limit) + extra_bars

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
        self._assert_raw_strategy_access_allowed("load_bars()")
        bars = self._runtime_load_feed(
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
        self._assert_raw_strategy_access_allowed("load_feed()")
        return self._runtime_load_feed(
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
        self._assert_raw_strategy_access_allowed("feed_series()")
        frame = self._runtime_load_feed(name)
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
        self._assert_raw_strategy_access_allowed("align_series()")
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

    def latest_decision_trace(self) -> list[dict]:
        """Return the most recent decision-context read trace."""
        if self._last_decision_context is None:
            return []
        return self._last_decision_context.trace_entries()

    def _assert_raw_strategy_access_allowed(self, method_name: str) -> None:
        if self._decision_surface_guard:
            raise RuntimeError(
                f"{method_name} is not available inside compute_decisions(); "
                "read market data through DecisionContext instead."
            )

    def compute_decisions(self, ctx: DecisionContext) -> DecisionDraft:
        """Preferred branch contract.

        Subclasses should return ``ctx.decisions(next_position)``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_decisions(ctx)."
        )

    def compute_signals(
        self,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        """Legacy signal contract kept during the rollout."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_signals(). "
            "Prefer compute_decisions(ctx) for new branch engines."
        )

    def get_latest_signal(self) -> dict:
        """Return the most recent effective and intended position."""
        compiled = self.compute_runtime_output()
        if len(compiled.decision_index) == 0:
            return {"position": 0.0, "next_position": 0.0, "date": "not-run"}
        return {
            "position": float(compiled.positions[-1]),
            "next_position": float(compiled.next_position[-1]),
            "date": str(compiled.decision_index[-1].date()),
        }

    def get_paper_signal(self, *, as_of=None) -> dict:
        """Return the next position decided at the close of ``as_of``.

        Engines can override this for a cheaper incremental path. The default
        implementation re-runs the engine up to ``as_of`` and returns the last
        compiled ``next_position``.
        """
        compiled = self.compute_runtime_output(end=as_of)
        if len(compiled.decision_index) == 0:
            return {"next_position": 0.0, "date": "not-run"}
        return {
            "next_position": float(compiled.next_position[-1]),
            "date": str(compiled.decision_index[-1].date()),
        }

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
            explicit = _selected_input_assets(self.research_branch_spec())
            selected = explicit
            if not selected:
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
        refs = coerce_graph_node_refs([item], extra_roles=[fallback_role])
        if not refs:
            return
        ref = refs[0]
        record = combined.setdefault(
            ref.node_id,
            {
                "node_id": ref.node_id,
                "ticker": ref.asset,
                "field": ref.field,
                "discovery_roles": set(),
            },
        )
        record["discovery_roles"].update(role for role in ref.roles if role)

    for item in discovery.get("parents") or []:
        remember(item, "parent")
    for item in discovery.get("blanket_new") or []:
        remember(item, "blanket")
    for item in discovery.get("children") or []:
        remember(item, "child")

    return [
        {
            "node_id": node_id,
            "ticker": payload["ticker"],
            "field": payload["field"],
            "discovery_roles": sorted(payload["discovery_roles"]),
        }
        for node_id, payload in sorted(combined.items())
    ]


def _explicit_candidates(branch_spec: dict) -> list[dict]:
    selected = branch_spec.get("selected_inputs") or []
    candidates: list[dict] = []
    for ref in coerce_graph_node_refs(selected):
        if not ref.asset:
            continue
        candidates.append(
            {
                "ticker": ref.asset,
                "field": ref.field,
                "node_id": ref.node_id,
                "discovery_roles": list(ref.roles) or ["selected"],
            }
        )
    return candidates


def _selected_input_assets(branch_spec: dict) -> list[str]:
    explicit = branch_spec.get("selected_inputs")
    if explicit:
        return [ref.asset for ref in coerce_graph_node_refs(explicit)]
    return []


def _research_target_node(context: dict | None) -> str | None:
    runtime_profile = ((context or {}).get("_runtime_profile") or {}) if isinstance(context, dict) else {}
    discovery = ((context or {}).get("discovery") or {}) if isinstance(context, dict) else {}
    branch_spec = ((context or {}).get("branch_spec") or {}) if isinstance(context, dict) else {}
    refs = coerce_graph_node_refs(
        [
            runtime_profile.get("target_node"),
            branch_spec.get("target_node"),
            discovery.get("target_node"),
            runtime_profile.get("target"),
            branch_spec.get("target_asset"),
            branch_spec.get("target"),
            discovery.get("ticker"),
        ]
    )
    return refs[0].node_id if refs else None


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
