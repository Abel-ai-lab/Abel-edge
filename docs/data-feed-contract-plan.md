# Data Feed Contract Plan

## Goal

Establish a framework-owned data loading and alignment contract so every
external input that enters `compute_signals()` is normalized, validated, and
auditable before strategy logic uses it.

This plan is intentionally modeled on the strongest part of mature backtesting
frameworks such as Backtrader: external data must first become a
framework-managed feed before it is allowed to participate in signal
generation.

The target outcome is:

- `Abel-edge` owns data semantics, time semantics, alignment semantics, and
  signal-output semantics
- third-party strategies keep freedom over features, signal logic, and sizing
- contract violations fail early at runtime with actionable exceptions
- the safe path is also the shortest path for strategy authors

## Problem

Today `Abel-edge` already owns part of the primary-price path:

- `StrategyEngine.load_bars()` routes through framework loaders
- `normalize_bars()` parses timestamps as UTC-aware datetimes
- `ledger.py` writes and deduplicates trade logs using UTC parsing

However, the framework does **not** yet own all external inputs used by a
strategy. Auxiliary series such as:

- volume
- cross-asset prices
- macro series
- factor series
- external CSV/API time series

can still be fetched and aligned outside the framework contract.

That leaves a failure mode where:

1. the engine's main dates are UTC-aware
2. an auxiliary series is naive or otherwise contract-incompatible
3. the strategy does a direct `reindex(...)` or date comparison
4. pandas silently misaligns instead of raising
5. the final `(positions, dates, prices)` tuple still looks structurally valid
6. drift only appears later in results, paper trading, or parity analysis

This plan closes that gap by making framework-managed feeds the only supported
way to bring external data into engine logic.

## In Scope

This plan should deliver:

1. a canonical daily datetime contract for engine-side data
2. feed declarations for non-primary external inputs
3. adapter interfaces that normalize external data into framework feed schemas
4. runtime validation for feed shape, timestamps, and alignment semantics
5. engine helper APIs that make safe data access easy
6. signal-output runtime validation before backtest/paper execution
7. tests and docs that make the supported runtime path explicit and easy to use

## Out Of Scope

This plan must not include:

- a full Backtrader-style line engine
- broker/order/fill simulation redesign
- intraday/multi-timeframe redesign in the first pass
- live trading provider failover chains
- factor-research orchestration outside strategy execution
- validation-metric redesign

This is a data-contract refactor, not a complete engine rewrite.

## Guiding Principles

1. Framework owns input contract; strategies own alpha logic.
2. Fail fast on data-contract violations.
3. Keep strategy-facing APIs pandas-friendly and array-first.
4. Preserve the simple `compute_signals() -> (positions, dates, prices)`
   contract.
5. Make the primary safe path shorter than ad hoc custom I/O.
6. Narrow framework promises to what can actually be enforced.
7. Start with a strict and well-defined `daily` profile before expanding
   further.
8. Prefer runtime contract enforcement over static policing of arbitrary user
   code.

## Responsibility Boundary

### `Abel-edge` Framework Responsibilities

The framework should own:

- config schema for declared feeds
- external-data adapter interface
- feed normalization and timestamp parsing
- canonical datetime contract
- feed alignment helpers
- engine output validation
- trade-log persistence contract
- test fixtures and docs for supported data access patterns
- agent-facing examples and ergonomics that make the supported path the default

### Third-Party Strategy Responsibilities

Strategy code should own:

- feature engineering on already-normalized feed data
- signal logic
- sizing logic
- research-specific transformations and filters
- optional custom `get_paper_signal()` behavior

### Unsupported Strategy Responsibilities

These should no longer be supported as strategy-local responsibilities:

- deciding whether external timestamps are naive or aware
- hand-rolling feed normalization semantics in strategy code
- direct `reindex(...)` of raw auxiliary series against strategy dates
- custom trade-log writing or date deduplication rules

Direct external I/O inside strategy code may still be possible in Python, but
it is outside the framework's correctness guarantees. The supported production
path is: declare feeds, load through framework helpers, align through framework
helpers, finalize through framework helpers.

## Target End State

After this plan lands, the execution path should become:

```text
strategies.yaml
  -> config.py
  -> feed declarations
  -> data adapters
  -> feed normalization + validation
  -> StrategyEngine helper APIs
  -> engine.compute_signals()
  -> signal contract validation
  -> backtest kernel / paper path
  -> ledger.py
  -> validate / dashboard
```

External inputs should enter engine logic only through framework-managed feed
access.

## Canonical Contracts

### Daily Datetime Contract

The first implementation pass should support a single canonical `daily`
contract:

- type: `pd.DatetimeIndex`
- timezone: UTC-aware only
- normalization: every timestamp normalized to `00:00:00+00:00`
- order: strictly increasing
- uniqueness: no duplicate timestamps within a feed/symbol series
- missing values: explicit and local to the feed; alignment policy must be
  chosen by the caller, not guessed silently

Naive datetimes are not allowed in the supported daily runtime contract.

For framework-owned file adapters such as CSV loaders, naive source timestamps
may be accepted as an input format and interpreted as UTC during normalization.
The contract above applies after loader normalization, inside engine runtime.

### Feed Kinds

The framework should initially support two feed kinds.

#### `bars`

Canonical columns:

- required: `timestamp`, `symbol`, `close`
- optional: `open`, `high`, `low`, `volume`, and other explicitly requested
  numeric fields

Use cases:

- primary tradeable asset
- external cross-asset prices
- multi-column market-bar inputs where `close` is part of the contract

#### `series`

Canonical columns:

- required: `timestamp`, `value`
- optional: `symbol`

Use cases:

- volume overlays
- macro series
- scalar factor series
- extracted single-field market inputs such as volume or a parent close series
- external indicator time series already computed outside the engine

### Signal Output Contract

Before any run or paper execution proceeds, `compute_signals()` output must
satisfy:

- `positions`, `dates`, and `prices` have identical lengths
- `dates` is a UTC-aware `DatetimeIndex`
- `dates` is strictly increasing with no duplicates or `NaT`
- `positions` and `prices` are numeric and finite
- daily-profile dates are normalized to midnight UTC

This contract is necessary but not sufficient. It catches malformed engine
outputs, while feed/alignment checks catch earlier semantic problems.

## Proposed Config Surface

The target config model should stop hard-coding a closed set of data-source
types inside framework core. Instead, `Abel-edge` should define a stable feed
contract and resolve provider-specific behavior through adapters.

Example:

```yaml
settings:
  price_data:
    default_adapter: abel
    default_timeframe: 1d
  data_contract:
    profile: daily
  data_adapters:
    imports:
      - trading_internal.data_adapters.market_data

strategies:
  - id: aapl_v2
    name: "AAPL V2"
    asset: AAPL
    color: "#2563EB"
    engine: strategies.aapl_v2.engine
    trade_log: data/trade_log_aapl_v2.csv
    price_data:
      adapter: abel
      symbol: AAPL
    feeds:
      aapl_volume:
        kind: series
        adapter: internal_cache_bars
        symbol: AAPL
        field: volume
```

Rules:

- `primary` remains user-implicit and continues to derive from the strategy
  asset when `load_bars()` is used
- `primary` is still a reserved framework-owned feed name
- every non-primary external dependency must still be declared under `feeds`
- undeclared feed access still fails
- the target contract should use `adapter`, not `source`, as the first-class
  config field
- `source` may be accepted only as a temporary migration alias if needed, but
  it is **not** part of the target design

This plan intentionally does **not** require the framework to statically prove
that no arbitrary Python I/O exists anywhere in strategy code. Instead, it
requires that the supported runtime path for production strategies goes through
declared feeds, registered adapters, and framework-managed helpers.

## Final Design Decisions

The following points are intentionally fixed in this plan to keep the
implementation simple and unambiguous.

### 1. `primary` Is User-Implicit But Internally Explicit

Strategy authors keep the simple current experience:

- `asset` + `price_data` define the primary data source
- `self.load_bars()` remains the default path for the tradeable asset

Internally, the framework should normalize every strategy config into an
explicit feed map that always includes a synthesized `primary` descriptor.

Consequence:

- the framework does not maintain separate logic for "primary data" versus
  "other feeds"
- `load_bars()` is ergonomic sugar for `load_feed("primary")`
- `feeds.primary` remains reserved and user-defined overrides are rejected

### 2. The Framework Owns Contract; Adapters Own Provider-Specific Fetching

`Abel-edge` should define what a valid feed looks like, but it should not own
every possible provider integration in core.

Consequence:

- framework core owns adapter registration and dispatch
- framework core may ship a **small** set of built-in adapters such as `abel`
  and `csv`
- third parties may register adapters such as `fmp`, `duckdb`, `parquet`, or
  repo-local cache adapters without waiting for framework-core changes
- provider-specific auth, caching, and raw column mapping live in adapters, not
  in strategy code

### 3. No Configurable `strict` Flag In The First Pass

The daily contract should be strict by definition.

Consequence:

- no `settings.data_contract.strict`
- no compatibility mode for invalid runtime datetimes
- contract violations fail with explicit runtime exceptions

File-backed adapters may accept naive source timestamps as an input format and
interpret them as UTC during normalization, but the runtime contract remains
strictly UTC-aware after adapter normalization.

### 4. `feed_series(...)` Is The Primary Public API For Single-Vector Inputs

For declared feeds that conceptually become a single aligned vector inside the
engine, strategy code should use `feed_series(...)`.

Consequence:

- `feed_series(...)` is the high-level default for auxiliary inputs
- `align_series(...)` is a lower-level helper for derived or already
  materialized in-memory series
- agent-authored strategies should prefer `feed_series(...)` unless they are
  aligning a series that was computed from already framework-managed data

### 5. Volume-And-Similar Inputs Use `series`, Not `bars`

To avoid ambiguity, `bars` and `series` should be separated by shape and intent.

Consequence:

- `bars` means close-bearing market bars
- `series` means one aligned vector such as volume, macro, factor, or a single
  extracted field
- volume-only auxiliary dependencies should be declared as `series` feeds
- an adapter may fetch bars underneath and derive a `series`, but the engine
  only sees the declared feed kind

## Adapter Registry Design

### Goal

Allow `Abel-edge` to stay simple and strongly constrained while still letting
third parties extend data-access behavior without modifying framework core.

### Registry Responsibilities

Framework core should provide:

- adapter registration
- adapter lookup
- loading of adapter import modules from config
- deterministic failure when a declared adapter is missing

Third parties should provide:

- adapter implementation objects
- any provider-specific config parsing
- provider-specific auth and caching behavior

### Suggested Adapter Request Shape

The adapter boundary should be narrow and provider-agnostic.

```python
@dataclass
class FeedLoadRequest:
    adapter: str
    kind: str
    symbol: str | None
    field: str | None
    timeframe: str | None
    start: object | None
    end: object | None
    limit: int | None
    profile: str
    options: dict[str, object]
    strategy_id: str | None
    feed_name: str
```

```python
class DataFeedAdapter(Protocol):
    def load(self, request: FeedLoadRequest) -> pd.DataFrame: ...
```

Adapter return requirements:

- `bars` adapters return at least `timestamp`, `symbol`, `close`
- `series` adapters return at least `timestamp`, `value`
- timestamps may arrive as strings, naive datetimes, or aware datetimes
- framework normalization converts adapter output into the canonical runtime
  contract before engine code uses it

### Built-In Versus Third-Party Adapters

Built-in adapters should stay deliberately small:

- `abel`
- `csv`

Everything else should be expected to live behind third-party registration, for
example:

- `fmp`
- `internal_cache_bars`
- `duckdb`
- `parquet`
- `vendor_x_api`

This keeps framework maintenance bounded while preserving extensibility.

## Proposed Engine APIs

The framework should keep the public engine surface small and centered on
`StrategyEngine`.

### Continue Supporting

- `self.load_bars(...)`

This remains the ergonomic path for the primary feed.

### Add

- `self.load_feed(name, *, start=None, end=None, limit=None, fields=None) -> pd.DataFrame`
- `self.feed_series(name, field="close", *, align_to=None, method=None) -> pd.Series`
- `self.align_series(series, dates, *, method="ffill", allow_gaps=True) -> pd.Series`
- `self.finalize_signals(positions, dates, prices) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]`

Recommended semantics:

- `load_bars()` becomes sugar for `load_feed("primary")`
- `feed_series(...)` is the default public path for declared single-vector
  inputs
- `align_series(...)` is the lower-level helper for derived or already
  materialized in-memory series
- `finalize_signals(...)` validates the signal output contract before returning

### Example Engine Shape

```python
class MyEngine(StrategyEngine):
    def compute_signals(self):
        bars = self.load_bars()
        target = bars[bars["symbol"] == self.context["asset"]].copy()
        dates = pd.DatetimeIndex(target["timestamp"])
        prices = target["close"].astype(float).to_numpy()

        ext_vol = self.feed_series(
            "aapl_volume",
            align_to=dates,
            method="ffill",
            allow_gaps=False,
        )

        positions = build_positions(prices, ext_vol)
        return self.finalize_signals(positions, dates, prices)
```

This keeps strategy code vectorized and familiar while forcing external inputs
through adapter-backed framework normalization and alignment.

## Proposed Internal Modules

Keep the implementation narrow and explicit.

### `causal_edge/engine/feed_contract.py`

Owns:

- datetime normalization helpers
- feed-level validation
- contract-specific exception classes

Suggested exception types:

- `FeedContractError`
- `FeedNormalizationError`
- `FeedAlignmentError`
- `SignalContractError`

### `causal_edge/engine/adapter_registry.py`

Owns:

- adapter registration
- adapter lookup
- config-driven import loading
- adapter existence checks used during config normalization

### `causal_edge/engine/feed_loader.py`

Owns:

- declared-feed resolution
- building `FeedLoadRequest`
- invoking adapters
- applying framework normalization after adapter output returns

### `causal_edge/engine/signal_contract.py`

Owns:

- validation of `(positions, dates, prices)` output
- reusable helper called by both run and paper paths

### `causal_edge/engine/base.py`

Extends `StrategyEngine` to expose the safe data-access and output-finalization
APIs described above.

### `causal_edge/config.py`

Extends schema validation for:

- `settings.data_contract`
- `settings.data_adapters.imports`
- strategy-level `feeds`
- rejection of user-defined `feeds.primary`
- normalization of user config into an explicit internal `primary` descriptor

## Runtime Enforcement Points

### 1. Adapter Output Normalization Gate

Every adapter returns raw provider data, but framework normalization runs before
that data reaches engine logic.

For a `daily` profile, this gate should reject:

- duplicate timestamps inside a symbol series
- unsorted timestamps
- unsupported columns for the declared feed kind
- invalid timestamp values

Naive file-backed timestamps may be accepted **only** as adapter input and
normalized into UTC. Naive in-memory runtime series remain invalid.

### 2. Declared Feed Access Gate

If a strategy asks for `self.load_feed("aapl_volume")` and that feed is not
declared in config, the framework fails immediately.

### 3. Adapter Resolution Gate

If a strategy declares `adapter: internal_fmp_bars` and that adapter is not
registered, config loading or feed loading should fail deterministically before
strategy logic proceeds.

### 4. Alignment Gate

Any external series aligned to strategy dates must pass through
`align_series(...)`.

This gate should reject:

- naive-vs-aware mismatches
- mismatched profile semantics
- duplicate dates in the input series
- alignment requests against invalid strategy dates

`feed_series(..., align_to=dates)` should use this same alignment logic
internally.

### 5. Signal Output Gate

`run_one()` and `paper_run_one()` should validate `compute_signals()` output
before backtest math, paper carry logic, or ledger writes.

### 6. Persistence Gate

`ledger.py` continues to own CSV shape and UTC date deduplication, but it
should no longer be the first place where semantic datetime problems surface.

### 7. Runtime-Centric Enforcement Boundary

This plan does **not** attempt to exhaustively block arbitrary Python I/O via
static linting or AST guardrails.

Instead, the framework guarantee is:

- if a strategy uses declared feeds and registered adapters, the framework
  guarantees data, time, and alignment contracts
- if a strategy bypasses those APIs, the framework does not guarantee semantic
  correctness beyond the final runtime gates it can still enforce

## How This Plan Intercepts The Current Internal Failure Mode

The motivating issue looked like this:

1. base engine returns UTC-aware `dates`
2. wrapper uses an auxiliary series fetched outside the framework
3. wrapper directly calls `reindex(dates_idx)`
4. pandas silently misaligns

With the adapter-registry plan, that failure is intercepted in one of two
places:

### Preferred Path

The auxiliary data is declared as a feed, resolved through a registered
adapter, normalized by the framework, and aligned through `feed_series(...)`.

Result:

- adapter-specific fetching stays outside strategy code
- framework normalization enforces the daily runtime contract
- `feed_series(..., align_to=dates)` sees compatible semantics
- the misalignment never occurs silently

### Explicit Failure Path

If the strategy still constructs or imports a raw auxiliary `Series`, it must
pass through `align_series(...)` before use.

Result:

- if the raw series is naive while `dates` is UTC-aware, the framework raises
  `FeedAlignmentError`
- the engine fails before producing trade logs

## Evaluation Milestones

The success criterion for this plan is **not** merely that implementation work
lands. Each milestone must provide stronger, testable evidence that the target
data/time mismatch class cannot survive silently on the supported runtime path.

### Milestone A: Contract Closure

Question:

- has the framework defined a complete and internally consistent
  adapter-registry-based contract for engine-side external data?

Evidence required:

- canonical `daily` datetime contract is documented
- supported feed kinds are documented
- adapter-registry model is documented
- config surface for non-primary feeds is documented
- supported alignment path is documented
- signal-output contract is documented
- reserved and synthesized `primary` semantics are documented

### Milestone B: Regression Interception

Question:

- does the framework now fail on the motivating naive/aware auxiliary mismatch
  before downstream drift occurs?

Evidence required:

- a dedicated regression fixture reproduces the wrapper-style mismatch
- the fixture fails with a feed/alignment contract exception
- the failure happens before trade-log writing

### Milestone C: Supported Path Usability

Question:

- can authors write compliant strategies naturally through declared feeds and
  adapters without bespoke timezone bridges or data-alignment patches?

Evidence required:

- a simple primary-only strategy still runs unchanged
- a strategy using at least one auxiliary feed runs through the canonical feed
  APIs
- a volume-style auxiliary input works naturally as a declared `series` feed
- bundled examples demonstrate the supported pattern end to end

### Milestone D: Runtime Gate Coverage

Question:

- do runtime gates collectively cover the points where this class of bug can be
  introduced or leak downstream?

Evidence required:

- adapter resolution gate rejects missing adapters
- adapter normalization gate rejects malformed feed timestamps
- declared-feed access gate rejects undeclared non-primary inputs
- alignment gate rejects contract-incompatible auxiliary series
- signal-output gate rejects malformed engine outputs
- run/paper paths fail early and deterministically

### Milestone E: Third-Party Adapter Extensibility

Question:

- can a third-party project add a non-built-in data source without modifying
  `Abel-edge` core and still receive full framework contract enforcement?

Evidence required:

- a project-local adapter can be registered via config-driven imports
- a strategy can declare and use that adapter in `price_data` or `feeds`
- adapter output still passes through framework normalization and runtime gates

### Milestone F: Real-World Migration Validation

Question:

- after migration, is the remaining drift no longer attributable to this
  data/time mismatch class?

Evidence required:

- representative wrapper/composite strategies are migrated to the framework
  path
- repo-local timezone bridge patches are no longer required for migrated paths
- the historical mismatch class now raises or is prevented by adapter/feed/
  alignment gates
- any remaining parity gaps can be attributed to other causes such as provider
  data differences or accounting-contract differences

## Implementation Sequence

Keep the rollout phased so existing projects can migrate without ambiguity.

### Phase 1: Contract Foundation

Deliver:

1. `settings.data_contract` schema
2. strict daily datetime helpers
3. signal-output contract validator
4. `StrategyEngine.finalize_signals(...)`
5. `run` / `paper` integration with signal-output validation
6. `primary` synthesis and `feeds.primary` rejection

Behavior:

- signal output contract becomes enforceable immediately
- primary path still works

### Phase 2: Adapter Registry Core

Deliver:

1. `adapter_registry.py`
2. config-driven adapter import loading
3. built-in `abel` and `csv` adapters
4. `adapter`-based config normalization
5. deterministic missing-adapter failures

Behavior:

- framework core no longer depends on a closed hard-coded source enum
- built-in adapters remain minimal

### Phase 3: Feed APIs And Runtime Gates

Deliver:

1. `load_feed(...)`, `feed_series(...)`, and `align_series(...)`
2. adapter-backed feed loading
3. undeclared-feed access failures
4. adapter normalization and alignment regressions

Behavior:

- all supported external inputs now have a framework path
- strategies no longer need ad hoc auxiliary loaders on the supported path

### Phase 4: Example And Migration Path

Deliver:

1. docs update in `add-strategy.md`
2. at least one bundled feed-path example
3. agent-friendly example configuration using adapters
4. migration guidance for third-party adapter modules

### Phase 5: Real-World Validation

Deliver:

1. migrate representative internal wrapper/composite strategies
2. compare behavior before and after migration
3. attribute remaining drift explicitly

## Test Plan

At minimum, add tests for:

1. config normalization synthesizes an explicit internal `primary` descriptor
2. user-defined `feeds.primary` is rejected
3. missing adapters fail deterministically
4. third-party adapter imports can register project-local adapters
5. `load_feed()` fails for undeclared feeds
6. built-in `csv` adapter accepts naive file timestamps and normalizes them
7. framework normalization rejects duplicate timestamps
8. `feed_series(..., align_to=dates)` rejects naive/aware mismatches
9. `align_series(...)` preserves valid aligned daily series
10. signal-output validation rejects naive `dates`
11. signal-output validation rejects duplicate or unsorted dates
12. `run` path fails early on contract-violating engine output
13. `paper` path fails early on contract-violating engine output
14. a bundled example using declared feeds runs end to end
15. a third-party adapter-backed strategy receives the same runtime validation as
    built-in adapters
16. a wrapper-style regression fixture reproduces the motivating mismatch and
    proves the framework now fails at adapter/feed/alignment time instead of
    silently drifting

## Definition Of Done

This plan is complete when:

1. `Abel-edge` provides a supported adapter-registry-based framework path for
   every external input that enters `compute_signals()`
2. daily-profile feeds are normalized to a single UTC-aware datetime contract
3. `primary` is user-implicit, internally explicit, and never handled via a
   separate special-case model
4. direct alignment of raw auxiliary series is no longer the recommended path
5. `run` and `paper` fail early when engine output violates the signal contract
6. third parties can register adapters without modifying framework core
7. the framework docs show a single canonical way to load primary and auxiliary
   data through adapters
8. the motivating wrapper-style datetime mismatch is covered by regression and
   now fails at adapter/feed/alignment gates instead of surfacing only as drift

## Non-Goals For Review

During implementation review, avoid expanding this pass into:

- intraday bar iterator design
- portfolio dependency graphs
- event-driven callback architecture
- broker/fill simulation
- a universal provider abstraction that tries to encode every market-data
  vendor in core

If a change starts pulling in those concerns, it is out of scope for this pass.

## Recommended First Implementation Slice

The smallest valuable first slice under the adapter-registry plan is:

1. add `signal_contract.py`
2. add `StrategyEngine.finalize_signals(...)`
3. validate engine output in `trader.py`
4. add `feed_contract.py` with daily datetime helpers
5. add `adapter_registry.py`
6. change config normalization from `source` semantics to `adapter` semantics

This slice does not finish adapter-backed feed loading, but it creates the
contract and registration primitives needed for the broader migration.

## Current Status Under Adapter Registry Semantics

The adapter-registry version of this plan is now implemented on the branch and
should be treated as the current target architecture rather than a reset point.

### Implemented In Framework Core

- `feed_contract.py` enforces the runtime datetime/alignment contract
- `signal_contract.py` enforces validated strategy outputs before backtest/paper
- `adapter_registry.py` provides built-in plus project-local adapter registration
- config loading imports project-local adapters through `settings.data_adapters.imports`
- synthesized `primary` feed semantics keep the primary path implicit while
  reserving `feeds.primary`
- `feed_loader.py` normalizes adapter-loaded frames before strategy runtime sees
  them
- bundled `abel` and `csv` adapters both run through the same framework-owned
  contract gates

### Implemented In Docs, Examples, And Regression Coverage

- `examples/feed_overlay_demo/` demonstrates declared `bars` + `series` feeds
- runtime tests cover undeclared feeds, incompatible datetimes, alignment
  failures, and invalid signal outputs
- adapter-registry tests cover project-local adapter imports and execution
- config/runtime tests cover primary-feed synthesis and primary adapter option
  passthrough

### Effective Milestone Status

- Milestones A-E are satisfied by the framework implementation and regression
  coverage on this branch
- Milestone F is effective based on real migration evidence from
  `trading-internal`, even though operational reruns remain an ongoing activity

As of 2026-04-16, the migrated `trading-internal` run completed successfully for
all currently registered strategies. Backup comparison on the already-reviewed
set shows the framework change preserved the original signal path closely enough
to support the new contract model:

- `seven_comp`: overlap PnL delta `+0.011844`, PnL corr `0.997453`
- `dr_v2`: overlap PnL delta `+0.169192`, position corr `1.000000`, PnL corr
  `0.999711`
- `dual_resonance`: overlap PnL delta `+0.187342`, position corr `1.000000`,
  PnL corr `0.999717`

These results are the key proof point for Milestone F: the runtime contract and
adapter registry semantics fixed the original data-path inconsistency without
breaking the established strategy signal path.

### Remaining Follow-Through

- keep public docs aligned with adapter terminology instead of older
  csv-only/source-only wording
- continue migration verification as new downstream strategies are moved onto
  the adapter path
- optionally remove temporary `source` alias compatibility once downstream
  configs are fully migrated
