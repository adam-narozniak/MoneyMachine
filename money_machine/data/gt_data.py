"""
Module to fetch for Google Trends data and perform appropriate scaling.

Conventions:
    Google Trends Data is kept in pd.DataFrame objects where the index is dt.datetime.
"""
import datetime as dt

import numpy as np
import pandas as pd
from pytrends.request import TrendReq


def create_timeframe_from_datetime(start_date: dt.date, end_date: dt.date):
    return str(start_date) + " " + str(end_date)


def check_fetch_data_correctness(start_date: dt.date, end_date: dt.date, overlap: dt.timedelta,
                                 date_diff: dt.timedelta):
    if date_diff <= overlap or date_diff <= dt.timedelta(0):
        raise Exception(
            f"Difference between the end date: {end_date} and start date: {start_date} should be greater than "
            f"the overlap: {overlap} and grater than 0 but is: {date_diff}")


def create_pulling_periods(start_date: dt.date, end_date: dt.date, overlap: dt.timedelta, period: dt.timedelta,
                           overlapping=True):
    """Based on the given start and end date create period that will be used for data pulling"""
    date_diff = end_date - start_date
    check_fetch_data_correctness(start_date, end_date, overlap, date_diff)
    starts = []
    ends = []
    current_start_date = start_date
    current_end_date = start_date + period
    while current_end_date < end_date:
        starts.append(current_start_date)
        ends.append(current_end_date)
        current_start_date = current_start_date + period - overlap
        current_end_date = current_end_date + period - overlap
    if overlapping is False:
        current_start_date = ends[-1] + dt.timedelta(1)
        current_end_date = end_date
        starts.append(current_start_date)
        ends.append(current_end_date)
    else:
        current_start_date = end_date - period
        current_end_date = end_date
        starts.append(current_start_date)
        ends.append(current_end_date)
    assert (len(starts) == len(ends))
    return starts, ends


def pull_overlapping_daily_data(fetcher, kw_list: list[str], start_dates: list[dt.date], end_dates: list[dt.date]):
    """
    Fetches the data from the google trends using interface from Fetcher.
    Args:
        fetcher: an instance that inherits from Fetcher
        kw_list: search keywords list
        start_dates:
        end_dates:

    Returns:
        data for given periods concatenated together.

    TODO: so check that the timedelta is the same for every pull
    """
    result = pd.DataFrame()
    for pull_id, (current_start_date, current_end_date) in enumerate(zip(start_dates, end_dates)):
        timeframe = create_timeframe_from_datetime(current_start_date, current_end_date)
        new_data = fetcher.fetch_data(kw_list, timeframe)
        new_data["pull_id"] = pull_id
        new_data.set_index(["pull_id", new_data.index], inplace=True)
        result = pd.concat([result, new_data], axis=0)
    return result


def create_overlap_periods(pull_starts, pull_ends, overlap):
    """Lists are in reverse chronological order."""
    pull_starts, pull_ends = pull_starts[1:], pull_ends[:-1]
    pull_starts.reverse()
    pull_ends.reverse()
    for a, b in zip(pull_starts, pull_ends):
        assert (b - a) >= overlap
    return pull_starts, pull_ends


def denormalize_by_overlapping_periods(data, overlap_starts, overlap_ends):
    pull_id = data.iloc[-1].name[0]
    normalized_data = data.loc[pull_id].iloc[:, 0].astype(np.float32)
    for overlap_start, overlap_end in zip(overlap_starts, overlap_ends):
        pull_id -= 1
        new_data = data.loc[pull_id].iloc[:, 0]
        normalized_overlap = normalized_data.loc[pd.Timestamp(overlap_start):pd.Timestamp(overlap_end)]
        max_normalized_overlap = normalized_overlap.max()
        max_normalized_overlap_id = normalized_overlap[normalized_overlap == max_normalized_overlap].index.values[0]
        new_data_reference_point = new_data.loc[pd.Timestamp(max_normalized_overlap_id)]
        scaling_factor = float(max_normalized_overlap) / new_data_reference_point
        new_data = new_data * scaling_factor
        normalized_data = pd.concat(
            [new_data.loc[:max_normalized_overlap_id], normalized_data.loc[max_normalized_overlap_id:]], axis=0)
        normalized_data = normalized_data / normalized_data.max() * 100.
    return normalized_data


def denormalize_daily_by_weekly(daily_data, weekly_data):
    # cut weekly data so it's for the same period as daily data
    min_daily_date = min(daily_data.index)
    max_daily_date = max(daily_data.index)
    daily_adj_by_weekly = pd.concat([daily_data, weekly_data], axis=1).fillna(method="ffill")
    daily_adj_by_weekly.dropna(inplace=True)
    adjusted = daily_adj_by_weekly.iloc[:, 0] * daily_adj_by_weekly.iloc[:, 1]
    adjusted = adjusted / adjusted.max() * 100
    return adjusted


def denormalize_daily_with_overlapping_periods_by_weekly(data, weekly_data):
    pull_ids = np.unique(data.index.get_level_values(0))
    denormalized = pd.DataFrame()
    for pull_id in pull_ids:
        temp_data = data.loc[pull_id].iloc[:, 0]
        adj = denormalize_daily_by_weekly(temp_data, weekly_data)
        adj = adj.to_frame()
        adj["pull_id"] = pull_id
        adj = adj.set_index(["pull_id", adj.index])
        denormalized = pd.concat([denormalized, adj], axis=0)
    return denormalized


if __name__ == "__main__":
    print("hello word")
    pytrends = TrendReq(hl='en-US', tz=360, retries=3, backoff_factor=0.1)
    kw_list = ["Blockchain"]
    start_date = '2021-03-24'
    end_date = '2021-12-24'
    timeframe_date_to_date = start_date + " " + end_date
    pytrends.build_payload(kw_list, timeframe=timeframe_date_to_date)
