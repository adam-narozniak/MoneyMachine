import datetime as dt

import pandas as pd
import pandas_datareader.data as web


from money_machine.ta.features import a_d_oscillator, cci, larry_wiliams_R, momentum, moving_average, rsi, signal_macd, \
    stochastic_k_percent, stochastic_d_percent, weighted_moving_average


def load_data(tickers: list[str], start: dt.datetime, end: dt.datetime):
    return {ticker: web.DataReader(ticker, "yahoo", start, end) for ticker in tickers}


def add_ta_data(data, n, fncs: list = None):
    data = data.copy()
    if fncs is None:
        fncs = [a_d_oscillator, cci, larry_wiliams_R, momentum, moving_average, rsi, signal_macd, stochastic_k_percent,
                stochastic_d_percent, weighted_moving_average]
    for fnc in fncs:
        data = fnc(data, n)
    return data


def y_label_for_n_day_pred(data, n):
    """
    Creates labels for prediction of n days ahead.

    Args:
        data: data from which the labels will be created (has to have "Close" column)
        n: the number of days ahead for prediction
    Returns:
        labels, shape[0] == data.shape[0]
    """
    y = data["Close"]
    y = y.shift(-n)
    y.name = "y"
    return y


def append_y(data, n):
    data = data.copy()
    return pd.concat([data, y_label_for_n_day_pred(data, n)], axis=1)


def divide_X_y(data):
    return data.iloc[:, :-1], data.iloc[:, -1].to_frame()
