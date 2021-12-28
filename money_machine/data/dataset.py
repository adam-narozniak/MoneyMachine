import pandas_datareader.data as web
import datetime as dt
import pandas as pd
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


def generate_multitimestep_data(data, n_additional_days):
    """
    Gather n_days data as a single instance.

    It accomplishes that by shifting the whole data and concatenating it together.

    Args:
        data:
        n_additional_days: number of days to have in a single row

    Returns:
        multitimestep_data

    """
    # stack the data from left to right (on left the earliest one, then the newer)
    new_data = data.shift(n_additional_days)
    for shift in range(n_additional_days - 1, -1, -1):
        shifted_data = data.shift(shift)
        shifted_data.columns = [col + f"_{shift}" for col in shifted_data.columns]
        new_data = pd.concat([new_data, shifted_data], axis=1)
    # new_data = new_data.dropna(axis=0)
    return new_data


def reshape_to_multistep_data(multistep_data, n_additional_days):
    return multistep_data.reshape(multistep_data.shape[0], (n_additional_days + 1), -1)


def y_label_for_n_day_pred(data, n):
    """
    Creates labels for prediction of n days ahead.

    Args:
        data: data from which the labels will be created (has to have "Close" column)
        n: the number of days ahead for prediction
    Returns:
        labels, shape[0] == data.shape[0]
    """
    y = data["Close"].shift(-n)
    y.name = "y"
    return y


def append_y(data, n):
    data = data.copy()
    return pd.concat([data, y_label_for_n_day_pred(data, n)], axis=1)


def drop_nans(data):
    return data.dropna(axis=0)


def divide_test_train(data, date):
    data_train, data_test = data.loc[:pd.Timestamp(date)], data.loc[pd.Timestamp(date):]
    return data_train, data_test


def divide_X_y(data):
    return data.iloc[:, :-1], data.iloc[:, -1].to_frame()
