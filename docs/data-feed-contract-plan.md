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

Keep the current implicit primary-asset ergonomics, but add explicit feed
declarations for all non-primary external inputs.

Example:

```yaml
settings:
  price_data:
    default_source: abel
    default_timeframe: 1d
  data_contract:
    profile: daily

strategies:
  - id: aapl_v2
    name: "AAPL V2"
    asset: AAPL
    color: "#2563EB"
    engine: strategies.aapl_v2.engine
    trade_log: data/trade_log_aapl_v2.csv
    feeds:
      aapl_volume:
        kind: series
        source: abel
        symbol: AAPL
        field: volume
        timeframe: 1d
```

Rules:

- `primary` remains implicit and continues to derive from the strategy asset
  when `load_bars()` is used
- `primary` is a reserved framework-owned feed name; users do not declare it
  under `feeds`
- every non-primary external data dependency must be declared under `feeds`
- undeclared feed access should fail
- internally, config normalization should synthesize an explicit `primary` feed
  descriptor so the framework operates on one feed model instead of special
  cases
- this plan removes the idea of a configurable `strict` mode for the daily
  contract; the supported runtime path is strict by default

This plan intentionally does **not** require the framework to statically prove
that no arbitrary Python I/O exists anywhere in strategy code. Instead, it
requires that the supported runtime path for production strategies goes through
declared feeds and framework-managed helpers.

## Final Design Decisions

The following points are intentionally fixed in this plan to keep the
implementation simple and unambiguous.

### 1. `primary` Is User-Implicit But Internally Explicit

Strategy authors keep the simple current experience:

- `asset` + `price_data` define the primary data source
- `self.load_bars()` remains the default path for the tradeable asset

Internally, however, the framework should normalize every strategy config into
an explicit feed map that always includes a synthesized `primary` descriptor.

Consequence:

- the framework does not maintain separate logic for "primary data" versus
  "other feeds"
- `load_bars()` is just ergonomic sugar for `load_feed("primary")`
- `feeds.primary` is reserved and user-defined overrides are rejected

### 2. No Configurable `strict` Flag In The First Pass

The daily contract should be strict by definition.

Consequence:

- no `settings.data_contract.strict`
- no compatibility mode for naive datetimes on the supported runtime path
- contract violations fail with explicit runtime exceptions

If a future migration needs warnings or softer behavior, that can be designed
later as a separate change with a clear justification. It should not blur the
initial contract.

### 3. `feed_series(...)` Is The Primary Public API For Single-Vector Inputs

For declared feeds that conceptually become a single aligned vector inside the
engine, strategy code should use `feed_series(...)`.

Consequence:

- `feed_series(...)` is the high-level default for auxiliary inputs
- `align_series(...)` is a lower-level helper for derived or already
  materialized in-memory series
- agent-authored strategies should prefer `feed_series(...)` unless they are
  aligning a series that was computed from already framework-managed data

### 4. Volume-And-Similar Inputs Use `series`, Not `bars`

To avoid ambiguity, `bars` and `series` should be separated by shape and intent.

Consequence:

- `bars` means close-bearing market bars
- `series` means one aligned vector such as volume, macro, factor, or a single
  extracted field
- volume-only auxiliary dependencies should be declared as `series` feeds
- adapters may derive a `series` feed from an underlying bars source, but the
  engine only sees the declared feed kind

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
  inputs and returns a normalized, optionally aligned series
- `align_series(...)` is the lower-level helper for derived or already
  materialized in-memory series that still need contract-checked alignment
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
        )

        positions = build_positions(prices, ext_vol)
        return self.finalize_signals(positions, dates, prices)
```

This keeps strategy code vectorized and familiar while forcing external inputs
through framework-owned normalization and alignment.

## Proposed Internal Modules

Keep the implementation narrow and explicit.

### `causal_edge/engine/feed_contract.py`

Owns:

- feed dataclasses / schemas
- datetime normalization helpers
- feed-level validation
- contract-specific exception classes

Suggested exception types:

- `FeedContractError`
- `FeedNormalizationError`
- `FeedAlignmentError`
- `SignalContractError`

### `causal_edge/engine/feed_loader.py`

Owns:

- adapter registry
- declared-feed resolution
- built-in adapters for Abel bars and CSV sources
- helper functions used by `StrategyEngine`

### `causal_edge/engine/signal_contract.py`

Owns:

- validation of `(positions, dates, prices)` output
- reusable helper called by both run and paper paths

### `causal_edge/engine/base.py`

Extend `StrategyEngine` to expose the safe data-access and output-finalization
APIs described above.

### `causal_edge/config.py`

Extend schema validation for:

- `settings.data_contract`
- strategy-level `feeds`
- rejection of user-defined `feeds.primary`
- normalization of user config into an explicit internal `primary` descriptor

## Runtime Enforcement Points

### 1. Feed Adapter Normalization Gate

Every adapter must normalize its output before returning it to engine code.

For a `daily` profile, this gate should reject:

- naive timestamps
- duplicate timestamps inside a symbol series
- unsorted timestamps
- unsupported columns for the declared feed kind

This is the earliest and most reliable place to reject malformed external data.

### 2. Declared Feed Access Gate

If a strategy asks for `self.load_feed("aapl_volume")` and that feed is not
declared in config, the framework should fail immediately.

This is important for reproducibility and auditability.

### 3. Alignment Gate

Any external series aligned to strategy dates must pass through
`align_series(...)`.

This gate should reject:

- naive-vs-aware mismatches
- mismatched profile semantics
- duplicate dates in the input series
- alignment requests against invalid strategy dates

This gate directly addresses the motivating wrapper-strategy failure mode.

`feed_series(..., align_to=dates)` should use this same alignment logic
internally so the high-level API and lower-level helper share one contract.

### 4. Signal Output Gate

`run_one()` and `paper_run_one()` should validate `compute_signals()` output
before backtest math, paper carry logic, or ledger writes.

This prevents malformed outputs from leaking into downstream systems.

### 5. Persistence Gate

`ledger.py` should continue to own CSV shape and UTC date deduplication, but it
should no longer be the first place where semantic datetime problems surface.

The expectation is:

- trader path validates engine output first
- ledger path persists already-validated output

### 6. Runtime-Centric Enforcement Boundary

This plan does **not** attempt to exhaustively block arbitrary Python I/O via
static linting or AST guardrails.

Rationale:

- Python user code is too flexible for static bans to be complete
- static direct-I/O policing would increase maintenance cost and false-positive
  risk
- the expected primary authorship mode is agent-generated strategy code, where
  strong examples and strict runtime APIs provide better leverage than trying
  to enumerate every forbidden pattern

Instead, the framework guarantee should be stated precisely:

- if a strategy uses the supported feed APIs, the framework guarantees data,
  time, and alignment contracts
- if a strategy bypasses those APIs, the framework does not guarantee semantic
  correctness beyond the final runtime gates it can still enforce

This keeps the supported path strict without pretending the framework can fully
police arbitrary Python behavior.

## How This Plan Intercepts The Current Internal Failure Mode

The motivating issue looked like this:

1. base engine returns UTC-aware `dates`
2. wrapper uses a naive auxiliary series
3. wrapper directly calls `reindex(dates_idx)`
4. pandas silently misaligns

With this plan, that failure is intercepted in one of two places:

### Preferred Path

The auxiliary data is declared as a feed and loaded through the framework.

Result:

- the adapter normalizes timestamps to the daily UTC contract
- `feed_series(..., align_to=dates)` sees compatible semantics
- the misalignment never occurs

### Explicit Failure Path

If the strategy still constructs or imports a raw auxiliary `Series`, it must
pass through `align_series(...)` before use.

Result:

- if the raw series is naive while `dates` is UTC-aware, the framework raises
  `FeedAlignmentError`
- the engine fails at runtime before producing trade logs

This means the issue is surfaced at data-ingestion or alignment time rather
than after result drift is observed.

## Evaluation Milestones

The success criterion for this plan is **not** merely that a sequence of
implementation phases completes. Each step should be judged by whether it gives
the framework stronger, testable evidence that the motivating class of data/time
mismatch can no longer survive silently on the supported runtime path.

### Milestone A: Contract Closure

Question:

- has the framework defined a complete and internally consistent contract for
  engine-side external data?

Evidence required:

- canonical `daily` datetime contract is documented
- supported feed kinds are documented
- config surface for non-primary feeds is documented
- supported alignment path is documented
- signal-output contract is documented
- reserved and synthesized `primary` semantics are documented
- there is no ambiguity about `strict` behavior because the supported contract
  is strict by default

Failure mode if not achieved:

- implementation work will branch into incompatible assumptions about what the
  framework owns versus what strategy code owns

### Milestone B: Regression Interception

Question:

- does the framework now fail on the motivating naive/aware auxiliary mismatch
  before downstream drift occurs?

Evidence required:

- a dedicated regression fixture reproduces the wrapper-style mismatch
- the fixture fails with a feed/alignment contract exception
- the failure happens before trade-log writing

Failure mode if not achieved:

- the framework may have more APIs, but it has not yet solved the original
  problem

### Milestone C: Supported Path Usability

Question:

- can authors write compliant strategies naturally without bespoke timezone
  bridges or data-alignment patches?

Evidence required:

- a simple primary-only strategy still runs unchanged
- a strategy using at least one auxiliary feed runs through the canonical feed
  APIs
- a volume-style auxiliary input works naturally as a declared `series` feed
- bundled examples demonstrate the supported pattern end to end

Failure mode if not achieved:

- authors will continue bypassing the framework because the supported path is
  too awkward or incomplete

### Milestone D: Runtime Gate Coverage

Question:

- do runtime gates collectively cover the points where this class of bug can be
  introduced or leak downstream?

Evidence required:

- adapter normalization gate rejects malformed feed timestamps
- declared-feed access gate rejects undeclared non-primary inputs
- alignment gate rejects contract-incompatible auxiliary series
- signal-output gate rejects malformed engine outputs
- run/paper paths fail early and deterministically

Failure mode if not achieved:

- invalid data may still pass through one boundary and only appear later as
  unexpected PnL drift

### Milestone E: Real-World Migration Validation

Question:

- after migration, is the remaining drift no longer attributable to this
  data/time mismatch class?

Evidence required:

- representative wrapper/composite strategies are migrated to the framework
  path
- repo-local timezone bridge patches are no longer required
- the historical mismatch class now raises or is prevented by feed/alignment
  gates
- any remaining parity gaps can be attributed to different causes such as data
  source differences or accounting-contract differences

Failure mode if not achieved:

- the framework may pass synthetic tests while the original production problem
  remains unresolved in realistic strategies

## Implementation Sequence

Keep the rollout phased so existing projects can migrate without ambiguity.

### Phase 1: Foundation

Deliver:

1. `settings.data_contract` schema
2. strategy `feeds` schema
3. feed-contract module with daily-profile validation
4. signal-output contract validator
5. `StrategyEngine.finalize_signals(...)`
6. `run` / `paper` integration with signal-output validation
7. config normalization that synthesizes `primary` internally and rejects
   user-defined `feeds.primary`

Behavior:

- primary `load_bars()` path keeps working
- non-primary feed APIs are available
- signal output contract becomes enforceable immediately

### Phase 2: Feed Path Expansion

Deliver:

1. built-in adapters for `abel`, `csv bars`, and `csv series`
2. `load_feed(...)`, `feed_series(...)`, and `align_series(...)`
3. undeclared-feed access failures
4. migration of bundled examples to framework-managed auxiliary feeds where
   relevant

Behavior:

- all supported external inputs now have a framework path
- third-party strategies no longer need ad hoc auxiliary loaders

### Phase 3: Strictness Upgrade

Deliver:

1. docs update in `add-strategy.md` and examples
2. contract-focused tests for mismatch and failure cases
3. strict-by-default guidance for non-primary external data
4. agent-friendly examples that show the canonical supported path for
   auxiliary feeds

Behavior:

- framework-managed feed access becomes the canonical supported integration path
- agent-authored strategies can follow one obvious pattern for primary and
  auxiliary inputs

## Test Plan

At minimum, add tests for:

1. feed normalization rejects naive daily timestamps
2. feed normalization rejects duplicate timestamps
3. config normalization synthesizes an explicit internal `primary` descriptor
4. user-defined `feeds.primary` is rejected
5. `load_feed()` fails for undeclared feeds
6. `feed_series(..., align_to=dates)` rejects naive/aware mismatches
7. `align_series(...)` preserves valid aligned daily series
8. signal-output validation rejects naive `dates`
9. signal-output validation rejects duplicate or unsorted dates
10. `run` path fails early on contract-violating engine output
11. `paper` path fails early on contract-violating engine output
12. primary `load_bars()` remains backward-compatible for existing simple
    strategies while internally resolving through synthesized `primary`
13. a wrapper-style regression fixture reproduces the naive/aware auxiliary
    mismatch and proves the framework now fails at feed/alignment time instead
    of silently drifting

## Definition Of Done

This plan is complete when:

1. `Abel-edge` provides a supported framework path for every external input that
   enters `compute_signals()`
2. daily-profile feeds are normalized to a single UTC-aware datetime contract
3. `primary` is user-implicit, internally explicit, and never handled via a
   separate special-case model
4. direct alignment of raw auxiliary series is no longer the recommended path
5. `run` and `paper` fail early when engine output violates the signal contract
6. the framework docs show a single canonical way to load primary and auxiliary
   data
7. the motivating wrapper-style datetime mismatch is covered by a regression
   test and now fails at data or alignment gates instead of surfacing only as
   downstream drift

## Non-Goals For Review

During implementation review, avoid expanding this pass into:

- intraday bar iterator design
- portfolio dependency graphs
- event-driven callback architecture
- broker/fill simulation
- feed caching across unrelated subsystems

If a change starts pulling in those concerns, it is out of scope for this pass.

## Recommended First Implementation Slice

The smallest valuable first slice is:

1. add `signal_contract.py`
2. add `StrategyEngine.finalize_signals(...)`
3. validate engine output in `trader.py`
4. add `feed_contract.py` with daily datetime helpers
5. add `align_series(...)` with hard failure on naive/aware mismatch
6. update `docs/add-strategy.md` to recommend feed helpers over raw reindexing

This slice will not finish the entire feed-declaration redesign, but it creates
the enforcement primitives needed for the broader migration.

## Next Step After This Plan

Once this plan is reviewed and accepted, the next implementation pass should
start with Phase 1 and Phase 2 file scaffolding rather than trying to migrate
all existing strategy patterns in a single PR.

## Milestone 1 Execution Record

Milestone scope:

- complete the Phase 1 foundation slice
- make runtime signal/date contract enforceable in `run` and `paper`
- provide the first supported framework path for a declared auxiliary `series`
  feed

Implemented in `feat/runtime-data-feed-contract`:

- `settings.data_contract.profile` with strict `daily` validation
- strategy `feeds` schema validation
- reserved user-facing `feeds.primary` rejection plus internal synthesized
  `primary`
- `causal_edge/engine/feed_contract.py`
- `causal_edge/engine/signal_contract.py`
- `StrategyEngine.load_feed(...)`
- `StrategyEngine.feed_series(...)`
- `StrategyEngine.align_series(...)`
- `StrategyEngine.finalize_signals(...)`
- `run` / `paper` runtime signal-output validation
- declared `csv series` loading through framework-owned feed helpers

Validation run:

- date: 2026-04-16
- command:

```bash
.venv/bin/python -m pytest \
  tests/test_data_contract_runtime.py \
  tests/test_price_data.py \
  tests/test_execution_run.py \
  tests/test_paper.py \
  tests/test_engine_loader.py
```

- result: `21 passed in 0.62s`

Evidence captured:

- config normalization synthesizes internal `primary` and injects default
  `daily` contract
- user-defined `feeds.primary` is rejected at config load time
- alignment gate rejects naive auxiliary series against UTC-aware strategy dates
- `run` fails early on naive signal dates before normal execution proceeds
- `paper` fails early on naive signal dates before normal execution proceeds
- a declared auxiliary `csv series` feed can be loaded through
  `feed_series(...)`, aligned through the framework, and produces the expected
  positions `[0.2, 0.4, 0.6]`

Milestone conclusion:

- the framework now owns and enforces the first end-to-end runtime contract for
  daily dates, auxiliary-series alignment, and signal outputs
- the motivating mismatch class is no longer allowed to pass silently on the
  supported runtime path covered by this milestone
- next milestone should expand adapter/path coverage and tighten undeclared-feed
  usage boundaries

## Milestone 2 Execution Record

Milestone scope:

- expand the supported framework path from auxiliary `series` feeds to
  auxiliary `bars` feeds
- make undeclared-feed violations fail deterministically on `run` and `paper`
- update strategy-author guidance so the framework path is the shortest
  supported path

Implemented in `feat/runtime-data-feed-contract`:

- `run` / `paper` now wrap feed-contract violations raised during
  `engine.compute_signals()` as CLI-visible framework failures
- declared auxiliary `bars` feeds are covered by runtime regression tests
- undeclared feed access is covered by runtime regression tests on both `run`
  and `paper`
- `docs/add-strategy.md` now documents declared feeds, `load_feed(...)`,
  `feed_series(...)`, `align_series(...)`, and `finalize_signals(...)` as the
  canonical strategy path

Validation run:

- date: 2026-04-16
- command:

```bash
.venv/bin/python -m pytest \
  tests/test_data_contract_runtime.py \
  tests/test_execution_run.py \
  tests/test_paper.py \
  tests/test_engine_loader.py \
  tests/test_price_data.py
```

- result: `24 passed in 0.81s`

Evidence captured:

- a declared auxiliary `bars` feed can be loaded through the framework and
  consumed via `feed_series(..., field='close', align_to=dates)`, producing the
  expected positions `[0.2, 0.4, 0.6]`
- an undeclared feed dependency fails early on the `run` path with a
  deterministic strategy-scoped error
- an undeclared feed dependency fails early on the `paper` path with the same
  deterministic strategy-scoped error
- feed/alignment contract errors are now surfaced through the execution entry
  points rather than leaking as raw internal exceptions

Milestone conclusion:

- the framework path now covers both auxiliary `series` and auxiliary `bars`
  inputs in the tested daily-contract slice
- undeclared non-primary feed usage is no longer a soft convention; it is a
  runtime-enforced boundary on the supported execution path
- the next milestone can focus on broader migration examples or stricter
  coverage of remaining unsupported ad hoc patterns

## Milestone 3 Execution Record

Milestone scope:

- tighten runtime gate coverage so malformed primary/auxiliary feed timestamps
  are rejected instead of silently normalized away
- extend signal-contract regression coverage beyond the naive-date case
- record explicit evidence that bad input now fails at loader/gate boundaries

Implemented in `feat/runtime-data-feed-contract`:

- `normalize_bars(...)` now enforces the daily runtime contract instead of
  silently dropping duplicate timestamps
- framework-owned CSV loaders now interpret naive daily timestamps as UTC and
  standardize them into the runtime contract
- duplicate per-symbol bar timestamps now fail fast during normalization
- signal-output regression tests now cover unsorted dates and length mismatch
- `docs/add-strategy.md` now explains the distinction between file-backed input
  format and the runtime UTC-aware contract

Validation run:

- date: 2026-04-16
- command:

```bash
.venv/bin/python -m pytest \
  tests/test_data_contract_runtime.py \
  tests/test_price_data.py \
  tests/test_execution_run.py \
  tests/test_paper.py \
  tests/test_engine_loader.py
```

- result: `29 passed in 0.72s`

Evidence captured:

- `load_bars_from_csv(...)` accepts naive daily timestamps and normalizes them
  into UTC-aware runtime timestamps
- file-backed `series` feeds loaded through the framework accept naive daily
  timestamps and normalize them into UTC-aware runtime timestamps
- `normalize_bars(...)` still rejects duplicate per-symbol timestamps instead of
  deduplicating them implicitly
- runtime alignment still rejects naive in-memory auxiliary series
- `validate_signal_output(...)` rejects unsorted dates
- `validate_signal_output(...)` rejects mismatched `(positions, dates, prices)`
  lengths

Milestone conclusion:

- runtime gate coverage now distinguishes clearly between file-backed input
  normalization and in-memory runtime contract enforcement
- malformed primary bars, malformed auxiliary alignment inputs, undeclared feed
  access, and malformed signal tuples are all covered in the tested
  daily-contract slice
- the next milestone can move from synthetic gate coverage toward more realistic
  bundled example migration or wrapper/composite strategy validation
