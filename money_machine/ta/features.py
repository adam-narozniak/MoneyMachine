from money_machine.ta.utils import lowest_close_n, highest_close_n
import numpy as np
import pandas_ta as pta
from money_machine.ta.utils import check_args


def moving_average(data, n):
    check_args(moving_average, n)
    data["MA-14d"] = data["Close"].rolling(window=n).mean()
    return data


def weighted_moving_average(data, n):
    check_args(weighted_moving_average, n)
    weights = np.linspace(1, n, n)
    sum_weights = np.sum(weights)
    data[f"WMA-{n}d"] = data["Close"].rolling(window=n).apply(lambda x: np.sum(weights * x / sum_weights))
    return data


def momentum(data, n):
    check_args(momentum, n)
    data[f"Momentum-{n}days"] = data["Close"] - np.roll(data["Close"], n)
    data[f"Momentum-{n}days"][:n] = np.nan
    return data


def stochastic_k_percent(data, n):
    check_args(stochastic_k_percent, n)
    n_min = lowest_close_n(data, n)
    n_max = highest_close_n(data, n)
    numerator = data["Close"] - n_min
    denominator = n_max - n_min
    data[f"stochastic_k_percent-{n}d"] = numerator / denominator * 100
    return data


def stochastic_d_percent(data, n, n_stochastic_k_percents=None):
    check_args(stochastic_d_percent, n)
    if n_stochastic_k_percents is None:
        n_stochastic_k_percents = n
    data[f"stochastic_d_percent-{n}d"] = data[
        f"stochastic_k_percent-{n_stochastic_k_percents}d"].rolling(window=n).mean()
    return data


def rsi(data, n):
    check_args(rsi, n)
    data[f"rsi-{n}d"] = pta.rsi(data['Close'], length=n)
    return data


def signal_macd(data, n):
    check_args(signal_macd, n)
    macd = pta.macd(data["Close"], fast=12, slow=26, signal=n)
    signal = macd.iloc[:, -1]
    data[signal.name] = signal
    return data


def larry_wiliams_R(data, n):
    check_args(larry_wiliams_R, n)
    n_min = lowest_close_n(data, n)
    n_max = highest_close_n(data, n)
    numerator = n_max - data["Close"]
    denominator = n_max - n_min
    data[f"larry_wiliams_R-{n}d"] = numerator / denominator * 100
    return data


def a_d_oscillator(data, n):
    check_args(a_d_oscillator, n)
    nominator = data["High"] - data["Close"]
    denominator = data["High"] - data["Low"]
    data["a_d_oscillator"] = nominator / denominator
    return data


def cci(data, n):
    check_args(cci, n)
    data[f"cci-{n}d"] = pta.cci(data["High"], data["Low"], data["Close"], n)
    return data
