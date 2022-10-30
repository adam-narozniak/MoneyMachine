import datetime as dt
from typing import Union

import pandas as pd

HOURLY_DATE_STRING_FORMAT = "%Y-%m-%dT%H"


def create_timeframe_from_datetime(start_date: dt.date, end_date: dt.date):
    return str(start_date) + " " + str(end_date)


def create_hourly_timeframe(start_time: Union[dt.datetime, dt.date], end_time: Union[dt.datetime, dt.date]):
    """
    Creates timeframe that is in a format necessary for google trends query to get hourly data.
    When given dates instead of datetimes, start date has hour 0 and end date 23.
    Args:
        start_time:
        end_time:

    Returns:
        String in the format necessary for google trends query.
    """
    if isinstance(start_time, dt.date) and isinstance(end_time, dt.date):
        start_time = dt.datetime.combine(start_time, dt.time(0))
        end_time = dt.datetime.combine(end_time, dt.time(23))
    elif isinstance(start_time, dt.datetime) and isinstance(end_time, dt.datetime):
        pass
    timeframe = start_time.strftime(HOURLY_DATE_STRING_FORMAT) + " " + end_time.strftime(HOURLY_DATE_STRING_FORMAT)
    return timeframe


def transform_hourly_to_daily(hourly_data: pd.DataFrame, amount: str = "all"):
    """
    Averages weekly hourly data into daily.
    Args:
        hourly_data: data with MultiIndex ("pull_id", "date") when amount is "all" or
            Index ("date") when amount is "single"
        amount: {"all"|"single"} usage described above

    Returns:

    TODO: fix the 'all' for the name and type of index

    """
    if amount == "single":
        hourly_data = hourly_data.iloc[:, 0]
        dates = pd.to_datetime(hourly_data.index.get_level_values(0).values).date
        hourly_data = hourly_data.to_frame().set_index(dates, append=True)
        hourly_data.index = hourly_data.index.set_names("day", level=1)
        daily_data = hourly_data.groupby(level=1).mean()
        daily_data.index.name = "date"
        daily_data.index = pd.to_datetime(daily_data.index)
    elif amount == "all":
        hourly_data = hourly_data.iloc[:, 0]
        dates = pd.to_datetime(hourly_data.index.get_level_values(1).values).date
        hourly_data = hourly_data.to_frame().set_index(dates, append=True)
        hourly_data.index = hourly_data.index.set_names("day", level=2)
        daily_data = hourly_data.groupby(level=[0, 2]).mean()
    else:
        raise KeyError(f"The given argument for amount parameter: {amount} is not handled."
                       f"Please refer to the documentation for the possible options.")
    return daily_data


def determine_overlap(data1, data2):
    """
    Indexed of both data are sorted and there must be an overlap for this function to work.
    Args:
        data1:
        data2:

    Returns:

    """
    assert data1.shape[1] == 1
    assert data2.shape[1] == 1
    min_d1 = min(data1.index.date)
    min_d2 = min(data2.index.date)
    max_d1 = max(data1.index.date)
    max_d2 = max(data2.index.date)
    start = max(min_d1, min_d2)
    end = min(max_d1, max_d2)
    return start, end


def check_fetch_data_correctness(start_date: dt.date, end_date: dt.date, overlap: dt.timedelta,
                                 date_diff: dt.timedelta):
    if date_diff <= overlap or date_diff <= dt.timedelta(0):
        raise Exception(
            f"Difference between the end date: {end_date} and start date: {start_date} should be greater than "
            f"the overlap: {overlap} and grater than 0 but is: {date_diff}")


def merge_data(older: pd.DataFrame, newer: pd.DataFrame, strategy: str):
    """

    Args:
        older: data with single Index with dates in in it
        newer:
        strategy: {"full_older"|"full_newer"|"average"|"older_till_max"| "newer_from_max"}

    Returns:

    """
    if strategy == "full_older":
        older_last_id = older.index.date[-1]
        newer_first_id = older_last_id + dt.timedelta(1)
        newer_adjusted = newer.loc[newer_first_id:]
        merged = pd.concat([older, newer_adjusted], axis=0)
        return merged
    elif strategy == "average":
        return average_overlap(older, newer)
    else:
        raise KeyError("Other strategies under development")


def average_overlap(data1, data2):
    data = pd.concat([data1, data2], axis=0)
    data = data.reset_index()
    averaged_data = data.groupby("date").mean()
    return averaged_data
