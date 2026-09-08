"""Generic point-in-time scalar-series contract for auxiliary data."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from abel_edge.engine._point_in_time_json import (
    find_credential_path,
    finite_json_copy,
)
from abel_edge.engine.feed_contract import FeedContractError

POINT_IN_TIME_SERIES_CONTRACT = "abel-edge.point-in-time-series/v1"
_TOP_LEVEL_KEYS = {
    "contract",
    "series_id",
    "source",
    "schema",
    "materialization",
    "transforms",
    "availability",
    "provenance",
}
_REQUIRED_KEYS = _TOP_LEVEL_KEYS - {"transforms"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GRID_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$")


class PointInTimeSeriesContractError(FeedContractError):
    """Raised when a point-in-time series spec or frame is unsafe."""


@dataclass(frozen=True)
class PointInTimeSeriesSpec:
    """Validated, canonical definition of one scalar point-in-time series."""

    _payload: dict[str, Any]
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PointInTimeSeriesSpec":
        if not isinstance(value, Mapping):
            raise PointInTimeSeriesContractError("series_spec must be a mapping.")
        try:
            payload = finite_json_copy(value)
        except FeedContractError as exc:
            raise PointInTimeSeriesContractError(str(exc)) from exc
        _validate_spec(payload)
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return cls(_payload=payload, sha256=hashlib.sha256(canonical).hexdigest())

    @property
    def payload(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    @property
    def series_id(self) -> str:
        return str(self._payload["series_id"])

    @property
    def source_adapter(self) -> str:
        return str(self._payload["source"]["adapter"])

    @property
    def source_request(self) -> dict[str, Any]:
        return deepcopy(self._payload["source"]["request"])

    @property
    def schema(self) -> dict[str, Any]:
        return deepcopy(self._payload["schema"])

    def to_mapping(self) -> dict[str, Any]:
        return deepcopy(self._payload)


def normalize_point_in_time_series_frame(
    frame: pd.DataFrame,
    spec: PointInTimeSeriesSpec | Mapping[str, Any],
    *,
    name: str,
    assume_utc_for_naive: bool = False,
) -> pd.DataFrame:
    """Normalize adapter output, indexing strategy visibility by ``available_at``."""

    resolved = (
        spec if isinstance(spec, PointInTimeSeriesSpec) else PointInTimeSeriesSpec.from_mapping(spec)
    )
    observed_source_receipt = str(frame.attrs.get("source_receipt_sha256") or "")
    schema = resolved.schema
    event_field = str(schema["event_time_field"])
    value_field = str(schema["value_field"])
    availability = resolved.payload["availability"]
    required = {event_field, value_field}
    if availability["mode"] == "explicit":
        required.add(str(schema["available_at_field"]))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PointInTimeSeriesContractError(
            f"{name} is missing point-in-time fields: {missing}."
        )

    source = frame.reset_index(drop=True)
    normalized = pd.DataFrame(index=source.index)
    normalized["event_time"] = _utc_index(
        source[event_field],
        name=f"{name}.{event_field}",
        assume_utc_for_naive=assume_utc_for_naive,
    )
    normalized["available_at"] = _availability_index(
        source,
        schema=schema,
        availability=availability,
        event_time=normalized["event_time"],
        name=name,
        assume_utc_for_naive=assume_utc_for_naive,
    )
    normalized["value"] = pd.to_numeric(source[value_field], errors="coerce")
    if normalized["value"].isna().any() or not np.isfinite(normalized["value"]).all():
        raise PointInTimeSeriesContractError(f"{name}.value must contain only finite numbers.")

    revision_field = schema.get("revision_id_field")
    if revision_field:
        if revision_field not in source.columns:
            raise PointInTimeSeriesContractError(
                f"{name} is missing revision field '{revision_field}'."
            )
        normalized["revision_id"] = source[revision_field].astype(str)

    normalized = normalized.sort_values(["available_at", "event_time"]).reset_index(drop=True)
    if normalized["available_at"].duplicated().any():
        raise PointInTimeSeriesContractError(
            f"{name} contains duplicate available_at timestamps for a scalar series."
        )
    _validate_materialized_grid(normalized, resolved.payload["materialization"], name=name)
    normalized.insert(0, "timestamp", normalized["available_at"])
    normalized.attrs["series_id"] = resolved.series_id
    normalized.attrs["series_spec_sha256"] = resolved.sha256
    normalized.attrs["source_receipt_sha256"] = (
        observed_source_receipt
        or resolved.payload["provenance"]["source_receipt_sha256"]
    )
    return normalized


def assert_point_in_time_adapter_identity(
    frame: pd.DataFrame,
    spec: PointInTimeSeriesSpec | Mapping[str, Any],
    *,
    name: str,
    verify_source_receipt: bool = True,
) -> None:
    """Fail closed when adapter data or materialization identity drifts."""

    resolved = (
        spec if isinstance(spec, PointInTimeSeriesSpec) else PointInTimeSeriesSpec.from_mapping(spec)
    )
    expected = str(resolved.payload["provenance"]["source_receipt_sha256"])
    observed = str(frame.attrs.get("source_receipt_sha256") or "")
    if verify_source_receipt and not observed:
        raise PointInTimeSeriesContractError(
            f"{name} adapter did not return a source receipt."
        )
    if verify_source_receipt and observed != expected:
        raise PointInTimeSeriesContractError(
            f"{name} source receipt mismatch: expected {expected}, observed {observed}."
        )
    observed_spec = str(frame.attrs.get("series_spec_sha256") or "")
    if not observed_spec:
        raise PointInTimeSeriesContractError(
            f"{name} adapter did not return its materialized spec identity."
        )
    if observed_spec != resolved.sha256:
        raise PointInTimeSeriesContractError(
            f"{name} spec identity mismatch: expected {resolved.sha256}, "
            f"observed {observed_spec}."
        )


def _validate_spec(payload: dict[str, Any]) -> None:
    unknown = sorted(set(payload) - _TOP_LEVEL_KEYS)
    missing = sorted(_REQUIRED_KEYS - set(payload))
    if unknown:
        raise PointInTimeSeriesContractError(f"series_spec has unknown keys: {unknown}.")
    if missing:
        raise PointInTimeSeriesContractError(f"series_spec is missing keys: {missing}.")
    if payload.get("contract") != POINT_IN_TIME_SERIES_CONTRACT:
        raise PointInTimeSeriesContractError(
            f"series_spec.contract must be '{POINT_IN_TIME_SERIES_CONTRACT}'."
        )
    if not str(payload.get("series_id") or "").strip():
        raise PointInTimeSeriesContractError("series_spec.series_id must be non-empty.")
    _validate_source(payload["source"])
    _validate_schema(payload["schema"], availability=payload["availability"])
    _validate_materialization(payload["materialization"])
    _validate_transforms(payload.get("transforms", []))
    _validate_availability(payload["availability"])
    _validate_provenance(payload["provenance"])


def _validate_source(source: Any) -> None:
    if not isinstance(source, dict):
        raise PointInTimeSeriesContractError("series_spec.source must be a mapping.")
    adapter = str(source.get("adapter") or "").strip().lower()
    if not adapter:
        raise PointInTimeSeriesContractError("series_spec.source.adapter must be non-empty.")
    request = source.get("request")
    if not isinstance(request, dict):
        raise PointInTimeSeriesContractError("series_spec.source.request must be a mapping.")
    unsafe = find_credential_path(request)
    if unsafe:
        raise PointInTimeSeriesContractError(
            f"series_spec must not embed credential field '{unsafe}'; use environment auth."
        )
    source["adapter"] = adapter


def _validate_schema(schema: Any, *, availability: Any) -> None:
    if not isinstance(schema, dict):
        raise PointInTimeSeriesContractError("series_spec.schema must be a mapping.")
    for key in ("event_time_field", "value_field"):
        if not str(schema.get(key) or "").strip():
            raise PointInTimeSeriesContractError(f"series_spec.schema.{key} must be non-empty.")
    if isinstance(availability, dict) and availability.get("mode") == "explicit":
        if not str(schema.get("available_at_field") or "").strip():
            raise PointInTimeSeriesContractError(
                "series_spec.schema.available_at_field is required for explicit availability."
            )


def _validate_materialization(value: Any) -> None:
    if not isinstance(value, dict):
        raise PointInTimeSeriesContractError(
            "series_spec.materialization must be a mapping."
        )
    if str(value.get("timezone") or "").upper() != "UTC":
        raise PointInTimeSeriesContractError(
            "series_spec.materialization.timezone must be 'UTC'."
        )
    if value.get("frequency") not in {"irregular", "calendar_day"}:
        raise PointInTimeSeriesContractError(
            "series_spec.materialization.frequency must be 'irregular' or 'calendar_day'."
        )
    if value.get("missing_policy") != "none":
        raise PointInTimeSeriesContractError(
            "series_spec.materialization.missing_policy must be 'none'."
        )
    if value.get("alignment_policy") not in {"asof", "native_only"}:
        raise PointInTimeSeriesContractError(
            "series_spec.materialization.alignment_policy must be 'asof' or "
            "'native_only'."
        )
    if value.get("frequency") == "calendar_day":
        _grid_offset(value.get("grid_time_utc"))


def _validate_transforms(value: Any) -> None:
    if not isinstance(value, list):
        raise PointInTimeSeriesContractError("series_spec.transforms must be a list.")
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not str(item.get("op") or "").strip():
            raise PointInTimeSeriesContractError(
                f"series_spec.transforms[{index}] must declare a non-empty op."
            )


def _validate_availability(value: Any) -> None:
    if not isinstance(value, dict):
        raise PointInTimeSeriesContractError("series_spec.availability must be a mapping.")
    mode = value.get("mode")
    if mode not in {"explicit", "calendar_days"}:
        raise PointInTimeSeriesContractError(
            "series_spec.availability.mode must be 'explicit' or 'calendar_days'."
        )
    if mode == "calendar_days":
        lag = value.get("lag_days")
        if not isinstance(lag, int) or isinstance(lag, bool) or lag < 0:
            raise PointInTimeSeriesContractError(
                "series_spec.availability.lag_days must be a nonnegative integer."
            )


def _validate_provenance(value: Any) -> None:
    if not isinstance(value, dict):
        raise PointInTimeSeriesContractError("series_spec.provenance must be a mapping.")
    if "source_receipt_sha256" not in value:
        raise PointInTimeSeriesContractError(
            "series_spec.provenance.source_receipt_sha256 is required."
        )
    for key, digest in value.items():
        text = str(digest or "")
        if key.endswith("_sha256") and (
            text != text.lower() or not _SHA256_RE.fullmatch(text)
        ):
            raise PointInTimeSeriesContractError(
                f"series_spec.provenance.{key} must be a lowercase SHA-256."
            )


def _availability_index(
    frame: pd.DataFrame,
    *,
    schema: dict[str, Any],
    availability: dict[str, Any],
    event_time: pd.Series,
    name: str,
    assume_utc_for_naive: bool,
) -> pd.Series:
    if availability["mode"] == "explicit":
        field = str(schema["available_at_field"])
        return pd.Series(
            _utc_index(
                frame[field],
                name=f"{name}.{field}",
                assume_utc_for_naive=assume_utc_for_naive,
            ),
            index=frame.index,
        )
    return event_time + pd.to_timedelta(int(availability["lag_days"]), unit="D")


def _utc_index(values, *, name: str, assume_utc_for_naive: bool) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, utc=False))
    except (TypeError, ValueError) as exc:
        raise PointInTimeSeriesContractError(f"{name} contains invalid timestamps.") from exc
    if len(index) == 0:
        return pd.DatetimeIndex([], tz="UTC")
    if index.tz is None:
        if not assume_utc_for_naive:
            raise PointInTimeSeriesContractError(f"{name} must be UTC-aware.")
        index = index.tz_localize("UTC")
    index = index.tz_convert("UTC")
    if index.hasnans:
        raise PointInTimeSeriesContractError(f"{name} contains NaT values.")
    return index


def _validate_materialized_grid(
    frame: pd.DataFrame,
    materialization: dict[str, Any],
    *,
    name: str,
) -> None:
    if materialization["frequency"] != "calendar_day" or frame.empty:
        return
    expected = _grid_offset(materialization["grid_time_utc"])
    observed = frame["event_time"] - frame["event_time"].dt.normalize()
    if not observed.eq(expected).all():
        raise PointInTimeSeriesContractError(
            f"{name}.event_time does not match grid_time_utc="
            f"{materialization['grid_time_utc']}."
        )


def _grid_offset(value: Any) -> pd.Timedelta:
    text = str(value or "")
    if not _GRID_TIME_RE.fullmatch(text):
        raise PointInTimeSeriesContractError(
            "series_spec.materialization.grid_time_utc must be HH:MM:SS."
        )
    try:
        offset = pd.to_timedelta(text)
    except (TypeError, ValueError) as exc:
        raise PointInTimeSeriesContractError(
            "series_spec.materialization.grid_time_utc must be HH:MM:SS."
        ) from exc
    if offset < pd.Timedelta(0) or offset >= pd.Timedelta(days=1):
        raise PointInTimeSeriesContractError(
            "series_spec.materialization.grid_time_utc must be within one day."
        )
    return offset
