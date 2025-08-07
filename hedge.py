from datetime import datetime, timedelta
import time
import requests
from io import StringIO

import numpy as np
import pandas as pd
import cvxopt
import cvxopt.solvers
from statistics import variance

## Loading historical prices ############################################################################################

def csvstr2df(string):
    file_ = StringIO(string)
    return pd.read_csv(file_, sep=",")

def datetime_to_timestamp(dt):
    return time.mktime(dt.timetuple()) + dt.microsecond / 1000000.0

def historical_prices(ticker_symbol):
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&includeAdjustedClose=true" % (
        ticker_symbol,
        int(datetime_to_timestamp(datetime.now() - timedelta(days=365))),
        int(datetime_to_timestamp(datetime.now()))
    )
    df = csvstr2df(requests.get(url).text)
    return df["Adj Close"]

## Helper functions #####################################################################################################

MARKET_DAYS_IN_YEAR = 252

def _truncate_quotes(quotes):
    truncated_quotes = {}
    for ticker in quotes:
        truncated_quotes[ticker] = quotes[ticker][-min(MARKET_DAYS_IN_YEAR, len(quotes[ticker])):]
    return truncated_quotes

def _remove_row(matrix, row):
    return np.vstack((matrix[:row], matrix[row + 1:]))

def _filter_negative_prices(price_matrix, ticker_map):
    # Remove stocks with any negative prices:
    index = 0
    while index < len(price_matrix):
        if any(value < 0.0 for value in price_matrix[index]):
            price_matrix = _remove_row(price_matrix, index)
            del ticker_map[index]
        else:
            index += 1

    return (price_matrix, ticker_map)

def _filter_duplicate_rows(price_matrix, ticker_map):
    # Remove duplicate rows:
    rowsEqual = lambda row1, row2: all(item == row2[index] for index, item in enumerate(row1))
    index1 = 0
    while index1 < len(price_matrix):
        index2 = index1 + 1
        while index2 < len(price_matrix):
            if rowsEqual(price_matrix[index1], price_matrix[index2]):
                price_matrix = _remove_row(price_matrix, index1)
                del ticker_map[index2]
            else:
                index2 += 1
        index1 += 1

    return (price_matrix, ticker_map)

def _filter_no_variance_rows(price_matrix, ticker_map):
    # Remove stocks with no variance:
    index = 0
    while index < len(price_matrix):
        if len(set(price_matrix[0])) == 1:
            price_matrix = _remove_row(price_matrix, index)
            del ticker_map[index]
        else:
            index += 1

    return (price_matrix, ticker_map)

def _filter_low_variance_rows(price_matrix, ticker_map):
    # Remove stocks stocks with low variance:
    VARIANCE_THRESHOLD = 0.1
    index = 0
    while index < len(price_matrix):
        if variance(price_matrix[index]) < VARIANCE_THRESHOLD:
            price_matrix = _remove_row(price_matrix, index)
            del ticker_map[index]
        else:
            index += 1

    return (price_matrix, ticker_map)

def _build_price_matrix(quotes, ticker):
    price_matrix = quotes[ticker]
    ticker_map = [ticker]

    for index, other_ticker in enumerate(quotes):
        if other_ticker != ticker:
            price_matrix = np.vstack((price_matrix, quotes[other_ticker]))
            ticker_map.append(other_ticker)

    price_matrix, ticker_map = _filter_negative_prices(price_matrix, ticker_map)
    price_matrix, ticker_map = _filter_duplicate_rows(price_matrix, ticker_map)
    price_matrix, ticker_map = _filter_no_variance_rows(price_matrix, ticker_map)
    price_matrix, ticker_map = _filter_low_variance_rows(price_matrix, ticker_map)

    return (price_matrix, ticker_map)

def _build_returns_matrix(price_matrix):
    returnsMatrix = []

    for row in price_matrix:
        returns = []
        for index in range(len(row) - 1):
            returns.append((row[index + 1] - row[index]) / row[index])
        returnsMatrix.append(returns)

    return returnsMatrix

def _minimize_portfolio_variance(returns_matrix):
    # Compose QP parameters:
    S = np.cov(returns_matrix)  # Sigma
    n = len(S) - 1
    P = np.vstack((np.hstack((2.0 * S[1:, 1:], np.zeros((n, n)))),
                      np.hstack((np.zeros((n, n)), 2.0 * S[1:, 1:]))))  # No negative here because -1 ^ 2 = 1
    q = np.vstack((2.0 * S[1:, 0:1],
                      -2.0 * S[1:, 0:1]))  # But this terms is linear so we do need the -1
    G = -np.eye(2 * n)
    h = np.zeros((2 * n, 1))
    A = np.ones((1, 2 * n))
    b = 1.0

    # Make QP parameters into CVXOPT matrices:
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)

    # Solve the QP:
    cvxopt.solvers.options["show_progress"] = False
    result = cvxopt.solvers.qp(P, q, G, h, A, b)
    weights = result["x"]

    return weights

def _filter_small_weights(weights):
    WEIGHT_THRESHOLD = 0.01
    for index, weight in enumerate(weights):
        if abs(weight) < WEIGHT_THRESHOLD:
            weights[index] = 0

    # We have to normalize weights after
    # discarding small ones above so that
    # they still sum to 1.
    weights = weights / sum(weights)

    return weights

def _compose_basket(weights, ticker_map, hedged_ticker_symbol):
    basket = {}

    for index in range(int(len(weights)/2)):
        pweight = weights[index]
        nweight = weights[int(len(weights) / 2) + index]
        weight = pweight - nweight
        if weight != 0 and ticker_map[index] != hedged_ticker_symbol:
            basket[ticker_map[index]] = float(weight) * -1.0

    return basket

## Public functions #####################################################################################################

def build_basket(hedged_ticker_symbol, basket_ticker_symbols):
    quotes = {ticker: historical_prices(ticker) for ticker in set(basket_ticker_symbols + [hedged_ticker_symbol])}
    quotes = _truncate_quotes(quotes)

    price_matrix, ticker_map = _build_price_matrix(quotes, hedged_ticker_symbol)
    returns_matrix = _build_returns_matrix(price_matrix)
    weights = _minimize_portfolio_variance(returns_matrix)
    weights = np.array(weights)
    weights = _filter_small_weights(weights)

    return _compose_basket(weights, ticker_map, hedged_ticker_symbol)

