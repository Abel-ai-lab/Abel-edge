# Abel Price API Context

This document aligns the planned Abel price API with `causal-edge` runtime needs.

## Goal

`causal-edge` should fetch real price bars from Abel by default, without exposing
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
- `causal-edge` currently uses Abel prod for both graph discovery and market data
- `causal-edge login --json --no-browser` emits a JSON handoff event first, then
  the final authorization result, which is the preferred flow for agent-driven
  environments

## Request Shape

```json
{
  "symbols": ["ETHUSD", "BTCUSD"],
  "start": "2023-01-01T00:00:00Z",
  "end": null,
  "timeframe": "1d",
  "limit": 600,
  "fields": ["open", "high", "low", "close", "volume"]
}
```

Notes:
- `symbols` are bare tickers like `ETHUSD`, not `ETHUSD.price`
- `timeframe` is currently expected to be `1d`
- `limit` is applied per symbol
- `fields` lets the API trim payloads later, but `causal-edge` currently expects
  at least `timestamp`, `symbol`, `close` in the response

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

## Runtime Expectations

`causal-edge` normalizes the response into a DataFrame with these standard columns:

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

## Why This Contract

- keeps `causal-edge` config simple
- keeps database schema hidden behind Abel
- supports multi-asset strategies and causal parent lookups
- matches the engine contract: strategies need aligned daily close series and may
  optionally use OHLCV fields later
