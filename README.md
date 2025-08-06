# Quantitative Hedging

A Python library for constructing hedging baskets to offset positions using historical price data and quadratic programming.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [License](#license)
- [Links](#links)

## Overview
The Quantitative Hedging repository provides an easy way to hedge a stock using a basket of other stocks which collectively behave as a hedge against the desired stock. It is intended for two types of users: (1) market makers who need to offset the risk derived from undesired inventory and (2) quantitative researchers who need to identify factors or replicate studies involving the performance of a security, portfolio, or hedge fund.

## Installation
Quantitative Hedging requires the following libraries:

- [pandas](https://pandas.pydata.org/)
- [cvxopt](https://cvxopt.org/)
- [numpy](https://numpy.org/)

Install these libraries with pip:

```
pip install -r requirements.txt
```

## Usage
This repository exposes a single public function `build_basket` (in `hedge.py`). The function builds a basket of stocks intended to hedge a desired stock.

```
from hedge import build_basket
build_basket(hedged_ticker_symbol, basket_ticker_symbols)
```

Arguments:

| Name | Type | Description | Optional | Example |
|------|------|-------------|---------|--------|
| `hedged_ticker_symbol` | `str` | Ticker symbol of the stock to hedge. | No | `"AAPL"` |
| `basket_ticker_symbols` | `list[str]` | Ticker symbols considered for the hedging basket. | No | `["GOOG", "MSFT", "NFLX", "AMZN", "FB"]` |

## Example
```
from hedge import build_basket

hedged_ticker_symbol = "AAPL"
basket_ticker_symbols = ["GOOG", "MSFT", "NFLX", "AMZN", "FB"]

print("Hedge for %s:" % hedged_ticker_symbol)
print(build_basket(hedged_ticker_symbol, basket_ticker_symbols))
```

This will produce output similar to:

```
{'AAPL': 0.2614353523521262, 'FB': 0.1921680128468791, 'AMZN': 0.5463966348009947}
```

A snippet like this can be incorporated in any Python application.

## License
Quantitative Hedging is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Links
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [CVXOPT](https://cvxopt.org/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)

