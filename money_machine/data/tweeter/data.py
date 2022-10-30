import datetime as dt

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from money_machine.data.tweeter.utils import parse_date_to_hashtag_count_list
from money_machine.data.utils import str_date_to_date


def pull_tweeter_hashtag_data(url: str, start: dt.datetime = None, end: dt.datetime = None):
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
            data_list = parse_date_to_hashtag_count_list(script_list)

    dates = []
    tweets = []
    for date, value in zip(data_list[0::2], data_list[1::2]):
        dates.append(date)
        tweets.append(str(value))
    date_format = "%Y/%m/%d"
    dates = [str_date_to_date(str_date, date_format) for str_date in dates]
    df = pd.DataFrame(np.array([dates, tweets]).T, columns=["date", "tweet-count"])
    df = df.set_index("date")
    df.loc[:, "tweet-count"] = df.loc[:, "tweet-count"].apply(lambda x: x if x != "null" else np.nan)
    df = df.astype(np.float64)
    return df.dropna(axis=0).loc[start:end]