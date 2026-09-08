# Abel Price API Context

This document aligns the planned Abel price API with `abel-edge` runtime needs.

## Goal

`abel-edge` should fetch real price bars from Abel by default, without exposing
database tables or SQL details in user config.

## Endpoint

- Current default environment: Abel prod
- Login endpoint: `GET https://api.abel.ai/echo/web/credentials/oauth/google/authorize/agent`
- CAP graph endpoint: `POST https://cap.abel.ai/api/cap`
- Market bars endpoint: `POST https://cap.abel.ai/api/market/day_bar`
- Auth header: `Authorization: Bearer <ABEL_API_KEY>`
- Override auth base with `ABEL_AUTH_BASE_URL=<custom_base>`
- Override base URL with `ABEL_CAP_BASE_URL=<custom_base>`

Notes:
- `abel-edge` currently uses Abel prod for both graph discovery and market data
- `abel-edge login --json --no-browser` emits a JSON handoff event first, then
  the final authorization result, which is the preferred flow for agent-driven
  environments

## Two Request Modes

`day_bar` has two mutually exclusive modes. Edge never converts one mode into
the other after a failed request.

| Mode | Identity | Intended use | Price adjustment |
| --- | --- | --- | --- |
| market bars | `symbols` | targets, ordinary OHLCV, and canonical close/volume nodes | provider front-adjusted market history |
| graph node | one exact `node_id` plus `shape=series` | canonical non-market V4 graph parents | node-native scalar values, no price adjustment |

Symbol mode uses UTC calendar dates because its bars have daily granularity.
Node mode keeps exact UTC timestamp bounds because those bounds filter
visibility times. The request must contain exactly one of `symbols` or
`node_id`; `node_ids` batching is not part of this interface.

### Symbol-mode request

```json
{
  "symbols": ["ETHUSD", "BTCUSD"],
  "start": "2023-01-01",
  "end": null,
  "timeframe": "1d",
  "limit": 600,
  "fields": ["open", "high", "low", "close", "volume"]
}
```

Notes:
- `symbols` are market-data symbols like `ETHUSD`, `605138.SS`, `9606.HK`,
  or `XFLI.TO`, not public graph IDs such as `ETHUSD.price`
- `start` and `end` are inclusive UTC calendar dates in `YYYY-MM-DD` form;
  Edge converts datetime-like caller values to their UTC calendar date before
  transport
- exchange suffixes are part of the market symbol and must not be stripped or
  interpreted as graph-node field suffixes
- `timeframe` is currently expected to be `1d`
- `limit` is applied per symbol
- `fields` lets the API trim payloads later, but `abel-edge` currently expects
  at least `timestamp`, `symbol`, `close` in the response
- A-share and Hong Kong price history returned by this mode is front-adjusted
  by the provider. Edge does not apply a second adjustment.

### Node-mode request

```json
{
  "node_id": "<exact registered CausalNodeV4 id>",
  "shape": "series",
  "start": "2023-01-01T00:00:00Z",
  "end": null,
  "limit": 600
}
```

Notes:
- the canonical ID is opaque and is sent byte-for-byte; public ticker aliases
  and symbol normalization do not apply
- CAP resolves the exact node internally and returns its scalar series; Edge
  does not receive or reproduce the node's selector or aggregation
- only non-market families use this mode; close/volume graph parents are
  deliberately rerouted before retrieval
- canonical close/volume nodes use symbol mode so A-share and Hong Kong
  corporate-action adjustment matches the market-data contract
- a missing node is an error; Edge does not retry it through `symbols`
- a logical scalar-series request that spans calendar years is sent as one
  non-cross-year CAP request per year; this is a transport partition only and
  Edge validates and receipts the merged logical series
- scalar-series pagination sends `cursor_date` from the previous response's
  `page.max_date`

## Response Shape

Preferred response:

```json
{
  "data": [
    {
      "timestamp": "2026-01-02T00:00:00Z",
      "symbol": "ETHUSD",
      "open": 3360.4,
      "high": 3412.0,
      "low": 3328.9,
      "close": 3398.1,
      "volume": 18234.2
    }
  ]
}
```

Also accepted by the current adapter:
- `{ "result": [...] }`
- `{ "data": { "bars": [...] } }`
- `{ "result": { "items": [...] } }`

Each returned row should represent one daily bar.

The scalar node response is:

```json
{
  "mode": "node_series",
  "node": {
    "node_id": "<exact registered CausalNodeV4 id>",
    "feature": "<value column>"
  },
  "data": [
    {
      "date": "2026-01-01",
      "timestamp": "2026-01-02T00:00:00Z",
      "value": 1.25
    }
  ],
  "page": {"limit": 100, "max_date": "2026-01-01", "has_more": true},
  "series": {"grain": "daily", "aggregation": "sum"}
}
```

The descriptor `node.node_id` must equal the requested ID. `timestamp` is the
earliest UTC visibility time; `event_time` is optional and defaults to that
timestamp. Edge rejects raw `mode=node_records`, non-finite values, duplicate
timestamps, non-UTC timestamps, identity drift, and stalled date cursors.
Calendar-year transport windows are validated against each row's source
`date`; a valid availability lag may place `timestamp` in the following
calendar year without moving the source observation outside its request.
For a bounded scalar-node request, Edge follows the complete date-cursor chain,
sorts the validated observations, and only then applies `limit` as a trailing-row
limit. A limited request therefore returns the most recent observations in the
window rather than the first page.

## Runtime Expectations

`abel-edge` normalizes the response into a DataFrame with these standard columns:

- `timestamp`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `volume`

Minimum required columns:
- `timestamp`
- `symbol`
- `close`

Runtime rules:
- timestamps must be parseable as UTC datetimes
- rows must be sortable by `symbol, timestamp`
- `(symbol, timestamp)` should be unique

Canonical node rows must have a unique UTC visibility timestamp. CAP applies
the canonical node's key/filter and aggregation inside its V4 registry. Edge
does not choose sum, mean, or last-row semantics on the consumer's behalf.

When an Abel canonical point-in-time feed declares an explicit `cache_root`,
Edge caches responses only after their effective upper bound has elapsed, under
the complete `series_spec` SHA-256. Reuse also requires the metadata's resolved
CAP endpoint and date-versus-timestamp bound semantics to match, plus a cached
requested range covering the new request. Open-ended, current-day, and future
requests bypass persisted cache reuse because a live response cannot prove
completeness. The cache file is content-hashed; a missing, edited, or
cross-endpoint entry is ignored and the canonical source is read again.
Prepare-time `source_start`, `source_end`, `source_limit`, and receipt provenance
do not constrain this live route. Node IDs are never used as filesystem paths.

## Why This Contract

- keeps `abel-edge` config simple
- keeps database schema hidden behind Abel
- supports multi-asset strategies and causal parent lookups
- matches the engine contract: strategies need aligned daily close series and may
  optionally use OHLCV fields later
