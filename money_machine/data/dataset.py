import datetime as dt

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup

from money_machine.data.utils import _parse_date_to_hashtag_count_list, _str_dates_to_dates
from money_machine.ta.features import a_d_oscillator, cci, larry_wiliams_R, momentum, moving_average, rsi, signal_macd, \
    stochastic_k_percent, stochastic_d_percent, weighted_moving_average


def load_data(tickers: list[str], start: dt.datetime, end: dt.datetime):
    return {ticker: web.DataReader(ticker, "yahoo", start, end) for ticker in tickers}


def load_saved_archive_data(path):
    """Loads data with index: 'pull_id', 'date'."""
    data = pd.read_csv(path, index_col=0)
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date", append=True)
    return data


def load_tweeter_hashtag_data(url: str, start: dt.datetime = None, end: dt.datetime = None):
    """
    Loads tweeter count data of the number of hashtags in a given day.

    Currently support only for the cryptocurrency data.

    Data comes from the: https://bitinfocharts.com/
    Code based on the: https://stackoverflow.com/a/59397210/11589429
    Args:
        url: urls to the website with this chart
        start:
        end:

    Returns:
        data to the count of the post with the twitter hashtag

    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    scripts = soup.find_all('script')
    data_list = None
    for script in scripts:
        script_list = script.text
        if 'd = new Dygraph(document.getElementById("container")' in script_list:
            script_list = '[[' + script_list.split('[[')[-1]
            script_list = script_list.split(']]')[0] + ']]'
            script_list = script_list.replace("new Date(", '').replace(')', '')
            data_list = _parse_date_to_hashtag_count_list(script_list)

    dates = []
    tweets = []
    for date, value in zip(data_list[0::2], data_list[1::2]):
        dates.append(date)
        tweets.append(str(value))
    dates = _str_dates_to_dates(dates)
    df = pd.DataFrame(np.array([dates, tweets]).T, columns=["date", "tweet-count"])
    df = df.set_index("date")
    df.loc[:, "tweet-count"] = df.loc[:, "tweet-count"].apply(lambda x: x if x != "null" else np.nan)
    df = df.astype(np.float64)
    return df.dropna(axis=0).loc[start:end]


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
