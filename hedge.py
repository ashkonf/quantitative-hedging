from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO
import logging
import time
from statistics import variance
from typing import TYPE_CHECKING, Dict, List, Tuple, cast

from cvxopt import matrix as cvx_matrix, solvers
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests

if TYPE_CHECKING:
    from cvxopt.base import matrix as CVXMatrix

MARKET_DAYS_IN_YEAR: int = 252

logger = logging.getLogger(__name__)


def csvstr2df(string: str) -> pd.DataFrame:
    """Convert a CSV string to a DataFrame."""

    file_: StringIO = StringIO(string)
    return pd.read_csv(file_, sep=",")


def datetime_to_timestamp(dt: datetime) -> float:
    """Return UNIX timestamp for a datetime."""

    return time.mktime(dt.timetuple()) + dt.microsecond / 1_000_000.0


def historical_prices(
    ticker_symbol: str, retries: int = 3, backoff: float = 0.3
) -> pd.Series:
    """Fetch adjusted closing prices from Yahoo Finance."""

    url: str = (
        "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s"
        "&interval=1d&events=history&includeAdjustedClose=true"
        % (
            ticker_symbol,
            int(datetime_to_timestamp(datetime.now() - timedelta(days=365))),
            int(datetime_to_timestamp(datetime.now())),
        )
    )

    for attempt in range(retries):
        try:
            logger.debug(
                "Fetching historical prices for %s (attempt %d/%d)",
                ticker_symbol,
                attempt + 1,
                retries,
            )
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            df: pd.DataFrame = csvstr2df(response.text)
            if "Adj Close" not in df.columns:
                raise ValueError("'Adj Close' column missing")
            return cast(pd.Series, df["Adj Close"])
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            logger.warning(
                "Attempt %d/%d to fetch %s failed: %s",
                attempt + 1,
                retries,
                ticker_symbol,
                exc,
            )
            if attempt == retries - 1:
                logger.error(
                    "Failed to retrieve historical prices for %s", ticker_symbol
                )
                raise RuntimeError(
                    f"Failed to retrieve historical prices for {ticker_symbol}"
                ) from exc
            time.sleep(backoff * 2**attempt)

    logger.error("Failed to retrieve historical prices for %s", ticker_symbol)
    raise RuntimeError(f"Failed to retrieve historical prices for {ticker_symbol}")


def _truncate_quotes(quotes: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """Limit each quote series to one year of data."""

    truncated_quotes: Dict[str, pd.Series] = {}
    for ticker in quotes:
        truncated_quotes[ticker] = cast(
            pd.Series,
            quotes[ticker][-min(MARKET_DAYS_IN_YEAR, len(quotes[ticker])) :],
        )
        logger.debug(
            "Truncated quotes for %s to %d entries",
            ticker,
            len(truncated_quotes[ticker]),
        )
    return truncated_quotes


def _remove_row(matrix: npt.NDArray[np.float64], row: int) -> npt.NDArray[np.float64]:
    """Return matrix with the specified row removed."""
    logger.debug("Removing row %d", row)
    return np.vstack((matrix[:row], matrix[row + 1 :]))


def _filter_negative_prices(
    price_matrix: npt.NDArray[np.float64], ticker_map: List[str]
) -> Tuple[npt.NDArray[np.float64], List[str]]:
    """Drop assets containing negative prices."""

    index: int = 0
    while index < len(price_matrix):
        if any(value < 0.0 for value in price_matrix[index]):
            logger.debug("Removing %s due to negative prices", ticker_map[index])
            price_matrix = _remove_row(price_matrix, index)
            del ticker_map[index]
        else:
            index += 1
    return (price_matrix, ticker_map)


def _filter_duplicate_rows(
    price_matrix: npt.NDArray[np.float64], ticker_map: List[str]
) -> Tuple[npt.NDArray[np.float64], List[str]]:
    """Remove duplicate rows from the price matrix."""

    def rows_equal(
        row1: npt.NDArray[np.float64], row2: npt.NDArray[np.float64]
    ) -> bool:
        """Return True if two rows are identical."""

        return all(item == row2[index] for index, item in enumerate(row1))

    index1: int = 0
    while index1 < len(price_matrix):
        index2: int = index1 + 1
        while index2 < len(price_matrix):
            if rows_equal(price_matrix[index1], price_matrix[index2]):
                logger.debug("Removing duplicate row for %s", ticker_map[index2])
                price_matrix = _remove_row(price_matrix, index1)
                del ticker_map[index2]
            else:
                index2 += 1
        index1 += 1
    return (price_matrix, ticker_map)


def _filter_no_variance_rows(
    price_matrix: npt.NDArray[np.float64], ticker_map: List[str]
) -> Tuple[npt.NDArray[np.float64], List[str]]:
    """Remove rows with no price variance."""

    index: int = 0
    while index < len(price_matrix):
        if len(set(price_matrix[index])) == 1:
            logger.debug("Removing %s due to no variance", ticker_map[index])
            price_matrix = _remove_row(price_matrix, index)
            del ticker_map[index]
        else:
            index += 1
    return (price_matrix, ticker_map)


def _filter_low_variance_rows(
    price_matrix: npt.NDArray[np.float64], ticker_map: List[str]
) -> Tuple[npt.NDArray[np.float64], List[str]]:
    """Remove rows whose variance is below the threshold."""

    variance_threshold: float = 0.1
    index: int = 0
    while index < len(price_matrix):
        row_values: List[float] = [float(x) for x in price_matrix[index]]
        if variance(row_values) < variance_threshold:
            logger.debug("Removing %s due to low variance", ticker_map[index])
            price_matrix = _remove_row(price_matrix, index)
            del ticker_map[index]
        else:
            index += 1
    return (price_matrix, ticker_map)


def _build_price_matrix(
    quotes: Dict[str, pd.Series], ticker: str
) -> Tuple[npt.NDArray[np.float64], List[str]]:
    """Construct the price matrix and ticker map."""

    price_matrix: npt.NDArray[np.float64] = quotes[ticker].to_numpy().reshape(1, -1)
    ticker_map: List[str] = [ticker]
    for index, other_ticker in enumerate(quotes):
        if other_ticker != ticker:
            price_matrix = np.vstack((price_matrix, quotes[other_ticker].to_numpy()))
            ticker_map.append(other_ticker)
    price_matrix, ticker_map = _filter_negative_prices(price_matrix, ticker_map)
    price_matrix, ticker_map = _filter_duplicate_rows(price_matrix, ticker_map)
    price_matrix, ticker_map = _filter_no_variance_rows(price_matrix, ticker_map)
    price_matrix, ticker_map = _filter_low_variance_rows(price_matrix, ticker_map)
    logger.debug("Built price matrix with shape %s for %s", price_matrix.shape, ticker)
    return (price_matrix, ticker_map)


def _build_returns_matrix(
    price_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute matrix of period-to-period returns."""

    return np.diff(price_matrix, axis=1) / price_matrix[:, :-1]


def _minimize_portfolio_variance(
    returns_matrix: npt.NDArray[np.float64],
) -> CVXMatrix:
    """Solve quadratic program to minimize portfolio variance."""

    s: npt.NDArray[np.float64] = np.cov(returns_matrix).astype(np.float64)
    n: int = len(s) - 1
    p: npt.NDArray[np.float64] = np.vstack(
        (
            np.hstack((2.0 * s[1:, 1:], np.zeros((n, n)))),
            np.hstack((np.zeros((n, n)), 2.0 * s[1:, 1:])),
        )
    )
    q: npt.NDArray[np.float64] = np.vstack((2.0 * s[1:, 0:1], -2.0 * s[1:, 0:1]))
    g: npt.NDArray[np.float64] = -np.eye(2 * n)
    h: npt.NDArray[np.float64] = np.zeros((2 * n, 1))
    a: npt.NDArray[np.float64] = np.ones((1, 2 * n))
    b: float = 1.0
    p_matrix = cvx_matrix(p)
    q_matrix = cvx_matrix(q)
    g_matrix = cvx_matrix(g)
    h_matrix = cvx_matrix(h)
    a_matrix = cvx_matrix(a)
    b_matrix = cvx_matrix(b)
    logger.debug("Solving quadratic program for %d assets", n)
    solvers.options["show_progress"] = False
    result = solvers.qp(p_matrix, q_matrix, g_matrix, h_matrix, a_matrix, b_matrix)
    weights = result["x"]
    logger.debug("Optimization returned weights %s", [float(x) for x in weights])
    return weights


def _filter_small_weights(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Zero out near-zero weights and renormalize."""

    weight_threshold: float = 0.01
    for index, weight in enumerate(weights):
        if abs(weight) < weight_threshold:
            weights[index] = 0
    weights = weights / sum(weights)
    logger.debug("Filtered small weights: %s", weights.tolist())
    return weights


def _compose_basket(
    weights: npt.NDArray[np.float64],
    ticker_map: List[str],
    hedged_ticker_symbol: str,
) -> Dict[str, float]:
    """Create a hedging basket from optimized weights."""

    basket: Dict[str, float] = {}
    for index in range(int(len(weights) / 2)):
        pweight: float = float(weights[index])
        nweight: float = float(weights[int(len(weights) / 2) + index])
        weight: float = pweight - nweight
        if weight != 0 and ticker_map[index] != hedged_ticker_symbol:
            basket[ticker_map[index]] = weight * -1.0
    logger.debug("Composed basket: %s", basket)
    return basket


def build_basket(
    hedged_ticker_symbol: str, basket_ticker_symbols: List[str]
) -> Dict[str, float]:
    """Construct a hedging basket for the target ticker."""

    logger.debug(
        "Building basket for %s with candidates %s",
        hedged_ticker_symbol,
        basket_ticker_symbols,
    )
    quotes: Dict[str, pd.Series] = {
        ticker: historical_prices(ticker)
        for ticker in set(basket_ticker_symbols + [hedged_ticker_symbol])
    }
    quotes = _truncate_quotes(quotes)
    price_matrix, ticker_map = _build_price_matrix(quotes, hedged_ticker_symbol)
    returns_matrix = _build_returns_matrix(price_matrix)
    weights = _minimize_portfolio_variance(returns_matrix)
    weights_array: npt.NDArray[np.float64] = np.array(weights)
    weights_array = _filter_small_weights(weights_array)
    basket = _compose_basket(weights_array, ticker_map, hedged_ticker_symbol)
    logger.debug("Built basket: %s", basket)
    return basket
