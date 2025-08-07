from datetime import datetime

import sys
from pathlib import Path

import cvxopt
import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge import (
    MARKET_DAYS_IN_YEAR,
    _build_price_matrix,
    _build_returns_matrix,
    _compose_basket,
    _filter_duplicate_rows,
    _filter_low_variance_rows,
    _filter_negative_prices,
    _filter_no_variance_rows,
    _filter_small_weights,
    _minimize_portfolio_variance,
    _remove_row,
    _truncate_quotes,
    build_basket,
    csvstr2df,
    datetime_to_timestamp,
    historical_prices,
)


def test_csv_and_timestamp() -> None:
    csv_data = "Adj Close\n1\n2\n"
    df = csvstr2df(csv_data)
    assert list(df["Adj Close"]) == [1, 2]
    ts = datetime_to_timestamp(datetime(1970, 1, 2))
    assert ts == pytest.approx(86400.0)


def test_historical_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        text = "Date,Adj Close\n2020-01-01,1\n2020-01-02,2\n"

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(requests, "get", lambda url: DummyResponse())
    series = historical_prices("AAPL")
    assert list(series) == [1, 2]


def test_historical_prices_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        text = ""

        def raise_for_status(self) -> None:  # pragma: no cover - test raising path
            raise requests.HTTPError("error")

    monkeypatch.setattr(requests, "get", lambda url: DummyResponse())
    with pytest.raises(RuntimeError):
        historical_prices("AAPL")


def test_truncate_and_remove_row() -> None:
    quotes = {"A": pd.Series(range(MARKET_DAYS_IN_YEAR + 10))}
    truncated = _truncate_quotes(quotes)
    assert len(truncated["A"]) == MARKET_DAYS_IN_YEAR

    matrix = np.array([[1, 2], [3, 4]])
    removed = _remove_row(matrix, 0)
    assert removed.tolist() == [[3, 4]]


def test_filters() -> None:
    matrix = np.array([[1, 2], [-1, 2]])
    tickers = ["A", "B"]
    filtered, tickers = _filter_negative_prices(matrix, tickers)
    assert filtered.tolist() == [[1, 2]] and tickers == ["A"]

    matrix = np.array([[1, 2], [1, 2], [3, 4]])
    tickers = ["A", "B", "C"]
    filtered, tickers = _filter_duplicate_rows(matrix, tickers)
    assert filtered.tolist() == [[1, 2], [3, 4]] and tickers == ["A", "C"]

    matrix = np.array([[1, 1, 1], [1, 2, 3]])
    tickers = ["A", "B"]
    filtered, tickers = _filter_no_variance_rows(matrix, tickers)
    assert filtered.tolist() == [[1, 2, 3]] and tickers == ["B"]

    matrix = np.array([[1, 1.0, 1.05], [1, 3, 5]])
    tickers = ["A", "B"]
    filtered, tickers = _filter_low_variance_rows(matrix, tickers)
    assert filtered.tolist() == [[1, 3, 5]] and tickers == ["B"]


def test_build_price_matrix_and_returns() -> None:
    quotes = {
        "A": pd.Series([1.0, 2.0, 3.0]),
        "B": pd.Series([1.0, 3.0, 1.0]),
    }
    price_matrix, ticker_map = _build_price_matrix(quotes, "A")
    assert ticker_map == ["A", "B"]
    returns_matrix = _build_returns_matrix(price_matrix)
    assert len(returns_matrix) == 2


def test_minimize_portfolio_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_qp(P, q, G, h, A, b):
        return {"x": [0.25, 0.25, 0.25, 0.25]}

    monkeypatch.setattr(cvxopt.solvers, "qp", fake_qp)  # type: ignore[attr-defined]
    returns = [[0.1, 0.2, 0.15], [0.05, 0.1, 0.0], [0.2, 0.1, 0.15]]
    weights = _minimize_portfolio_variance(returns)
    assert len(weights) == 4


def test_filter_small_weights_and_compose() -> None:
    weights = np.array([0.6, 0.02, 0.1, 0.28])
    filtered = _filter_small_weights(weights)
    assert pytest.approx(sum(filtered)) == 1.0

    basket = _compose_basket(filtered, ["A", "B"], "A")
    assert basket == {"B": pytest.approx(0.26)}


def test_filter_small_weights_threshold() -> None:
    weights = np.array([0.5, 0.005, 0.495, 0.0])
    filtered = _filter_small_weights(weights)
    assert filtered[1] == 0


def test_build_basket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hedge.historical_prices",
        lambda symbol: pd.Series([1, 2]) if symbol == "AAA" else pd.Series([2, 3]),
    )
    monkeypatch.setattr(
        "hedge._minimize_portfolio_variance",
        lambda rm: np.array([0.6, 0.4, 0.1, 0.2]),
    )
    basket = build_basket("AAA", ["BBB"])
    assert basket == {"BBB": pytest.approx(-0.15384615)}
