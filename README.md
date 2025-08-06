# Quantitative Hedging

A Python library for constructing hedging baskets to offset positions using historical price data and quadratic programming.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Linting](#linting)
- [Type Checking](#type-checking)
- [License](#license)
- [Links](#links)

## Overview
The Quantitative Hedging repository provides an easy way to hedge a stock using a basket of other stocks which collectively behave as a hedge against the desired stock. It is intended for two types of users: (1) market makers who need to offset the risk derived from undesired inventory and (2) quantitative researchers who need to identify factors or replicate studies involving the performance of a security, portfolio, or hedge fund.

## Installation
This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install all dependencies, including development tools, with:

```bash
uv sync --all-extras --dev
```

## Usage
This repository exposes a single public function `build_basket` (in `hedge.py`). The function builds a basket of stocks intended to hedge a desired stock.

```python
from hedge import build_basket
build_basket(hedged_ticker_symbol, basket_ticker_symbols)
```

Arguments:

| Name | Type | Description | Optional | Example |
|------|------|-------------|---------|--------|
| `hedged_ticker_symbol` | `str` | Ticker symbol of the stock to hedge. | No | `"AAPL"` |
| `basket_ticker_symbols` | `list[str]` | Ticker symbols considered for the hedging basket. | No | `["GOOG", "MSFT", "NFLX", "AMZN", "FB"]` |

## Testing
Run the test suite with full coverage using:

```bash
uv run pytest
```

## Linting
Format and lint the code with [ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff format
uv run ruff check .
```

## Type Checking
Static type checking is performed with [pyright](https://github.com/microsoft/pyright):

```bash
uv run pyright
```

## License
Quantitative Hedging is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Links
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [CVXOPT](https://cvxopt.org/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
