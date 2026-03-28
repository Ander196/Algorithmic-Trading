# CLAUDE.md

## Project Overview

Modular Python app for stock trading. Core features: market data ingestion, portfolio tracking, and strategy backtesting. Prioritize correctness and testability — financial logic must be deterministic.

## Structure

```
src/
  data/         # Market data fetching and normalization
  portfolio/    # Position tracking, P&L calculation
  backtesting/  # Strategy simulation engine and metrics
  strategies/   # Strategy implementations
  models/       # Shared domain types (Trade, Position, OHLCV, Signal)
  utils/        # Logging, config, date helpers
tests/
  unit/         # No I/O — mock all external calls
  integration/  # Gated with @pytest.mark.integration
  fixtures/     # Sample OHLCV data for tests
```

## Dev Commands

```bash
pip install -e ".[dev]"          # Install with dev dependencies
pytest tests/unit/               # Run unit tests
pytest -m integration            # Run integration tests
ruff check src/ tests/           # Lint
ruff format src/ tests/          # Format
mypy src/ --strict               # Type check
```

All three checks (ruff, mypy, pytest unit) must pass before committing.

## Architecture

- **Models are the source of truth.** All domain types live in `src/models/`. Never redefine them elsewhere.
- **Backtester is a pure function.** `run_backtest(data, strategy) -> BacktestResult` — no side effects, no global state.
- **Strategies implement a standard protocol.** All strategies expose `generate_signals(data: pd.DataFrame) -> list[Signal]`.
- **Data layer is stateless.** Fetchers accept config, return normalized domain objects. No internal state.
- **Config over hardcoding.** API keys and parameters come from env vars or `config/`. Never hardcode them.

## Code Conventions

- Python 3.11+. Use modern syntax (`X | Y` unions, `match`, etc.).
- Type annotations required on all public functions and class attributes.
- Use `decimal.Decimal` for monetary values — never `float` for money.
- Timestamps must be **timezone-aware**. Never use naive datetimes for market data.
- OHLCV columns: always lowercase `open`, `high`, `low`, `close`, `volume`.
- Raise domain-specific exceptions (`DataFetchError`, `InsufficientDataError`). Don't swallow errors silently.
- **Functions Naming** Use `camelCase`for functions and methods
- **Naming** Use clear, descriptive names for functions and classes