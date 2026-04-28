"""Decision-context authoring surface for agent-first branch engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd

from abel_edge.engine.runtime_contract import (
    DecisionDraft,
    DecisionContractError,
    ExecutionConstraints,
    RuntimeProfile,
    build_decision_draft,
)


@dataclass(frozen=True)
class DecisionTraceEntry:
    """Structured record of a strategy-visible runtime read."""

    surface: str
    feed: str
    field: str
    rows: int
    decision_time: str | None = None
    start: str | None = None
    end: str | None = None
    aligned_to_decision_index: bool = False


class DecisionContext:
    """Runtime-owned decision-time world exposed to ``compute_decisions``."""

    def __init__(
        self,
        engine,
        *,
        runtime_profile: RuntimeProfile,
        execution_constraints: ExecutionConstraints,
        start=None,
        end=None,
        limit: int | None = None,
    ) -> None:
        self.engine = engine
        self.runtime_profile = runtime_profile
        self.execution_constraints = execution_constraints
        self.start = start
        self.end = end
        self.limit = limit
        self.target = _DecisionTargetView(self)
        self._target_frame_cache: dict[tuple[str, ...], pd.DataFrame] = {}
        self._feed_frame_cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}
        self._trace: list[DecisionTraceEntry] = []

    def decision_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.target.series("close").index)

    def feed(self, name: str):
        return _DecisionFeedView(self, name)

    def input(self, name: str):
        return self.feed(name)

    def available_feeds(self) -> list[str]:
        feeds = (self.engine.context or {}).get("_feeds") or {}
        return sorted(str(name) for name in feeds.keys())

    def available_inputs(self) -> list[str]:
        return [name for name in self.available_feeds() if name != "primary"]

    def input_specs(self) -> list[dict[str, Any]]:
        feeds = (self.engine.context or {}).get("_feeds") or {}
        specs: list[dict[str, Any]] = []
        for name in self.available_inputs():
            cfg = dict(feeds.get(name) or {})
            cfg.setdefault("name", name)
            cfg.setdefault("default_field", self._default_feed_field(name))
            specs.append(cfg)
        return specs

    def inputs_frame(
        self,
        *names: str,
        mode: str = "asof",
        fields: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        selected = list(names) if names else self.available_inputs()
        if mode not in {"asof", "native"}:
            raise DecisionContractError("inputs_frame(mode=...) must be 'asof' or 'native'.")
        columns: list[pd.Series] = []
        for name in selected:
            field_name = (fields or {}).get(name)
            view = self.input(name)
            series = (
                view.asof_series(field_name)
                if mode == "asof"
                else view.native_series(field_name)
            )
            columns.append(series.rename(name))
        if not columns:
            index = self.decision_index() if mode == "asof" else pd.DatetimeIndex([])
            return pd.DataFrame(index=index)
        frame = pd.concat(columns, axis=1)
        if mode == "asof":
            frame = frame.reindex(self.decision_index())
        return frame

    def inspect_feed(self, name: str) -> dict[str, Any]:
        field = self._default_feed_field(name)
        frame = self._load_feed_frame(name, field)
        fields = [column for column in frame.columns if column not in {"timestamp", "symbol"}]
        first = pd.to_datetime(frame["timestamp"], utc=True).min() if not frame.empty else None
        last = pd.to_datetime(frame["timestamp"], utc=True).max() if not frame.empty else None
        return {
            "name": name,
            "field": field,
            "rows": int(len(frame)),
            "fields": fields,
            "first_timestamp": _to_trace_value(first),
            "last_timestamp": _to_trace_value(last),
        }

    def points(self) -> Iterator["DecisionPoint"]:
        index = self.decision_index()
        for idx, ts in enumerate(index):
            yield DecisionPoint(self, idx, ts)

    def decisions(self, next_position) -> DecisionDraft:
        return build_decision_draft(
            self.decision_index(),
            next_position,
            runtime_profile=self.runtime_profile,
        )

    def trace_entries(self) -> list[dict[str, Any]]:
        return [
            {
                "surface": item.surface,
                "feed": item.feed,
                "field": item.field,
                "rows": item.rows,
                "decision_time": item.decision_time,
                "start": item.start,
                "end": item.end,
                "aligned_to_decision_index": item.aligned_to_decision_index,
            }
            for item in self._trace
        ]

    def preview(self, *, limit: int = 5) -> list[dict[str, Any]]:
        close = self.target.series("close")
        preview = close.tail(limit)
        return [
            {"date": str(ts), "target_close": float(value)}
            for ts, value in preview.items()
        ]

    def sample_points(self, *, limit: int = 3) -> list[dict[str, Any]]:
        points = list(self.points())
        if not points:
            return []
        if len(points) <= limit:
            selected = points
        else:
            anchors = sorted({0, len(points) // 2, len(points) - 1})
            selected = [points[idx] for idx in anchors[:limit]]
        return [point.to_dict() for point in selected]

    def trace_point(self, date) -> dict[str, Any]:
        target_date = _as_utc_timestamp(date)
        for point in self.points():
            if point.decision_time() == target_date:
                return point.to_dict()
        raise DecisionContractError(
            f"DecisionContext has no decision point at {target_date.isoformat()}."
        )

    def _load_target_frame(self, *fields: str) -> pd.DataFrame:
        requested_fields = tuple(sorted(set(fields or ("close",))))
        if requested_fields not in self._target_frame_cache:
            frame = self.engine._runtime_load_bars(
                start=self.start,
                end=self.end,
                limit=self.limit,
                fields=list(requested_fields),
            )
            target = self.runtime_profile.target
            if target and "symbol" in frame.columns:
                frame = frame[frame["symbol"].astype(str).str.upper() == target].copy()
            if frame.empty:
                raise DecisionContractError(
                    "DecisionContext could not load target bars for the active runtime profile."
                )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.sort_values("timestamp").reset_index(drop=True)
            self._target_frame_cache[requested_fields] = frame
        return self._target_frame_cache[requested_fields]

    def _load_feed_frame(self, name: str, *fields: str) -> pd.DataFrame:
        requested_fields = tuple(sorted(set(fields or ("close",))))
        cache_key = (name, requested_fields)
        if cache_key not in self._feed_frame_cache:
            frame = self.engine._runtime_load_feed(
                name,
                start=self.start,
                end=self.end,
                limit=self.limit,
                fields=list(requested_fields),
            )
            if frame.empty:
                raise DecisionContractError(f"Feed '{name}' returned no rows for the active context.")
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.sort_values("timestamp").reset_index(drop=True)
            self._feed_frame_cache[cache_key] = frame
        return self._feed_frame_cache[cache_key]

    def _record_trace(
        self,
        *,
        surface: str,
        feed: str,
        field: str,
        rows: int,
        decision_time=None,
        start=None,
        end=None,
        aligned_to_decision_index: bool = False,
    ) -> None:
        self._trace.append(
            DecisionTraceEntry(
                surface=surface,
                feed=feed,
                field=field,
                rows=int(rows),
                decision_time=_to_trace_value(decision_time),
                start=_to_trace_value(start),
                end=_to_trace_value(end),
                aligned_to_decision_index=aligned_to_decision_index,
            )
        )

    def _default_feed_field(self, name: str) -> str:
        feed_cfg = ((self.engine.context or {}).get("_feeds") or {}).get(name) or {}
        explicit = str(feed_cfg.get("default_field") or "").strip().lower()
        if explicit:
            return explicit
        if str(feed_cfg.get("kind") or "").strip().lower() == "series":
            return "value"
        return "close"


class _DecisionTargetView:
    def __init__(self, ctx: DecisionContext) -> None:
        self.ctx = ctx

    def series(self, field: str = "close") -> pd.Series:
        frame = self.ctx._load_target_frame(field)
        if field not in frame.columns:
            raise DecisionContractError(f"Target bars do not expose field '{field}'.")
        series = pd.Series(
            frame[field].astype(float).to_numpy(),
            index=pd.DatetimeIndex(frame["timestamp"]),
            name=f"target.{field}",
        )
        self.ctx._record_trace(
            surface="target.series",
            feed=self.ctx.runtime_profile.target or "target",
            field=field,
            rows=len(series),
            aligned_to_decision_index=True,
        )
        return series


class _DecisionFeedView:
    def __init__(self, ctx: DecisionContext, name: str) -> None:
        self.ctx = ctx
        self.name = name

    def native_series(self, field: str | None = None) -> pd.Series:
        field_name = field or self.ctx._default_feed_field(self.name)
        frame = self.ctx._load_feed_frame(self.name, field_name)
        series = _frame_to_series(frame, field=field_name, feed_name=self.name)
        self.ctx._record_trace(
            surface="feed.native_series",
            feed=self.name,
            field=field_name,
            rows=len(series),
        )
        return series

    def asof_series(self, field: str | None = None) -> pd.Series:
        field_name = field or self.ctx._default_feed_field(self.name)
        native = self.native_series(field_name)
        aligned = _align_asof_to_index(native, self.ctx.decision_index())
        self.ctx._record_trace(
            surface="feed.asof_series",
            feed=self.name,
            field=field_name,
            rows=len(aligned),
            aligned_to_decision_index=True,
        )
        return aligned

    def interval_matrix(self, *args, **kwargs):
        raise NotImplementedError("interval_matrix() is not implemented in the V1 rollout.")


class DecisionPoint:
    """One legal decision-time point from a ``DecisionContext``."""

    def __init__(self, ctx: DecisionContext, index_position: int, timestamp: pd.Timestamp) -> None:
        self.ctx = ctx
        self.index_position = index_position
        self.timestamp = pd.Timestamp(timestamp)
        self.target = _PointTargetView(self)

    def decision_time(self) -> pd.Timestamp:
        return self.timestamp

    def prev_target_close(self) -> pd.Timestamp | None:
        if self.index_position <= 0:
            return None
        return self.ctx.decision_index()[self.index_position - 1]

    def feed(self, name: str):
        return _PointFeedView(self, name)

    def input(self, name: str):
        return self.feed(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_time": str(self.decision_time()),
            "prev_target_close": _to_trace_value(self.prev_target_close()),
            "target_close": float(self.target.history("close", bars=1).iloc[-1]),
        }


class _PointTargetView:
    def __init__(self, point: DecisionPoint) -> None:
        self.point = point

    def history(self, field: str = "close", *, bars: int | None = None) -> pd.Series:
        series = self.point.ctx.target.series(field)
        window = series.loc[series.index <= self.point.timestamp]
        if bars is not None:
            window = window.tail(int(bars))
        self.point.ctx._record_trace(
            surface="point.target.history",
            feed=self.point.ctx.runtime_profile.target or "target",
            field=field,
            rows=len(window),
            decision_time=self.point.timestamp,
        )
        return window


class _PointFeedView:
    def __init__(self, point: DecisionPoint, name: str) -> None:
        self.point = point
        self.name = name

    def history(self, field: str | None = None, *, bars: int | None = None) -> pd.Series:
        field_name = field or self.point.ctx._default_feed_field(self.name)
        native = self.point.ctx.feed(self.name).native_series(field_name)
        window = native.loc[native.index <= self.point.timestamp]
        if bars is not None:
            window = window.tail(int(bars))
        self.point.ctx._record_trace(
            surface="point.feed.history",
            feed=self.name,
            field=field_name,
            rows=len(window),
            decision_time=self.point.timestamp,
        )
        return window

    def between(self, start, end, *, field: str | None = None) -> pd.Series:
        field_name = field or self.point.ctx._default_feed_field(self.name)
        native = self.point.ctx.feed(self.name).native_series(field_name)
        start_ts = _as_utc_timestamp(start) if start is not None else native.index.min()
        end_ts = _as_utc_timestamp(end) if end is not None else self.point.timestamp
        window = native.loc[(native.index >= start_ts) & (native.index <= end_ts)]
        self.point.ctx._record_trace(
            surface="point.feed.between",
            feed=self.name,
            field=field_name,
            rows=len(window),
            decision_time=self.point.timestamp,
            start=start_ts,
            end=end_ts,
        )
        return window

    def asof(self, field: str | None = None):
        field_name = field or self.point.ctx._default_feed_field(self.name)
        history = self.history(field_name)
        if history.empty:
            return None
        self.point.ctx._record_trace(
            surface="point.feed.asof",
            feed=self.name,
            field=field_name,
            rows=1,
            decision_time=self.point.timestamp,
        )
        return history.iloc[-1]


def _frame_to_series(frame: pd.DataFrame, *, field: str, feed_name: str) -> pd.Series:
    if "value" in frame.columns and field == "value":
        values = frame["value"]
    elif field in frame.columns:
        values = frame[field]
    else:
        raise DecisionContractError(f"Feed '{feed_name}' does not expose field '{field}'.")
    return pd.Series(
        values.astype(float).to_numpy(),
        index=pd.DatetimeIndex(frame["timestamp"]),
        name=f"{feed_name}.{field}",
    )


def _align_asof_to_index(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    if series.empty or len(target_index) == 0:
        return pd.Series(dtype=float, index=target_index, name=series.name)
    expanded = series.reindex(series.index.union(target_index)).sort_index().ffill()
    aligned = expanded.reindex(target_index)
    aligned.name = series.name
    return aligned


def _to_trace_value(value) -> str | None:
    if value is None:
        return None
    return str(_as_utc_timestamp(value))


def _as_utc_timestamp(value) -> pd.Timestamp:
    ts = value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
