"""
Module to fetch for Google Trends data and perform appropriate scaling.

Conventions:
    Google Trends Data is kept in pd.DataFrame objects where the index is dt.datetime.
    If it is concatenated together for many different periods it has MultiIndex: "pull_id", "date".
"""
import datetime as dt

import numpy as np
import pandas
import pandas as pd
from pytrends.request import TrendReq

from money_machine.data.pytrends_fetcher import PytrendsFetcher


def create_timeframe_from_datetime(start_date: dt.date, end_date: dt.date):
    return str(start_date) + " " + str(end_date)


def check_fetch_data_correctness(start_date: dt.date, end_date: dt.date, overlap: dt.timedelta,
                                 date_diff: dt.timedelta):
    if date_diff <= overlap or date_diff <= dt.timedelta(0):
        raise Exception(
            f"Difference between the end date: {end_date} and start date: {start_date} should be greater than "
            f"the overlap: {overlap} and grater than 0 but is: {date_diff}")


def average_overlap(data1, data2):
    data = pd.concat([data1, data2], axis=0)
    data = data.reset_index()
    averaged_data = data.groupby("date").mean()
    return averaged_data


def create_pulling_periods(start_date: dt.date,
                           end_date: dt.date,
                           overlap: dt.timedelta,
                           period: dt.timedelta) -> tuple[list[dt.date], list[dt.date]]:
    """
    Based on the given start and end date create period that will be used for data pulling. It starts creating the
    intervals form the start (in chronological order).

    Start dates and ends dates can be thought of as a recursive sequences.
    start(0) and end(N) are given
    start(n) = n*period - n*overlap
    end(n) = (n+1)*period - n*overlap
    start(N) is not always calculated using this formula (in order to make the result more accurate)
    end(N) can't be calculated using that formula if it is not a perfect fit (period and overlap wise; end(N) had to
        satisfy the given formula which is rate)
    end(N) is calculated as end(N-1) - overlap, which is the same as start(N)

    Args:
        start_date: the fist date of the search of the first period
        end_date: the last date of the search of the last period
        overlap: number of days that two consecutive periods will have the same,
            especially 0 means that the periods won't have any days in common; it should be at least of the length of
            the smallest time acceptable in google trends for query
        period: length (in days) of the single period
    Note:
        The last period might have equal or smaller overlapping period. It's because the there are probably not discrete
         number of parts of periods with overlaps in the specified timeframe. The normal procedure is to decrease the
         last period.
    Returns:
        chronologically ordered lists of starts and ends
    """
    delta = overlap - dt.timedelta(1)
    period = period  # - dt.timedelta(1)
    period_minus_two_deltas = period - 2 * delta
    if period_minus_two_deltas < dt.timedelta(0):
        # minus ones here are for the human interpretability for words overlap and period
        raise Exception(f"The period should be greater or equal that 2*delta where delta = (overlap -1).\n"
                        f"The values provided are period: {period}, overlap: {overlap}.\n"
                        f"They  don't satisfy the condition.\n"
                        f"For given period: {period}, the biggest overlap which you can give as an argument is {period / 2 + dt.timedelta(1)}")
    date_diff = end_date - start_date
    check_fetch_data_correctness(start_date, end_date, delta, date_diff)
    starts = []
    ends = []
    current_start_date = start_date
    current_end_date = start_date + period
    while current_end_date < end_date + dt.timedelta(1):
        starts.append(current_start_date)
        ends.append(current_end_date)
        current_start_date = current_start_date + period - delta
        current_end_date = current_end_date + period - delta
    # add the last period only if it's needed, meaning when the last computed end date is not the same as end_date
    if current_end_date != end_date:
        # ends[-1] - delta is the same as the normal start would be at that time
        # it could be also left as current_start_date
        current_start_date = ends[-1] - delta
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

    TODO: so check that the timedelta is the same for every entry in each pull
    """
    result = pd.DataFrame()
    for pull_id, (current_start_date, current_end_date) in enumerate(zip(start_dates, end_dates)):
        timeframe = create_timeframe_from_datetime(current_start_date, current_end_date)
        new_data = fetcher.fetch_data(kw_list, timeframe)
        new_data["pull_id"] = pull_id
        new_data.set_index(["pull_id", new_data.index], inplace=True)
        result = pd.concat([result, new_data], axis=0)
    return result


def create_anchor_banks(gtab_fetcher, start_dates: list[dt.date], end_dates: list[dt.date]):
    for current_start_date, current_end_date in zip(start_dates, end_dates):
        timeframe = create_timeframe_from_datetime(current_start_date, current_end_date)
        gtab_fetcher.create_anchorbank(timeframe)


def create_overlap_periods(pull_starts: list[dt.date], pull_ends: list[dt.date], overlap: dt.timedelta):
    """Returned lists are in reverse chronological order."""
    pull_starts, pull_ends = pull_starts[1:], pull_ends[:-1]
    pull_starts.reverse()
    pull_ends.reverse()
    for a, b in zip(pull_starts, pull_ends):
        dates_diff = b - a + dt.timedelta(1)
        assert dates_diff >= overlap, f"The date {b} should be later than {a} by at least {overlap}, " \
                                      f"but is {dates_diff} instead"
    return pull_starts, pull_ends


def denormalize_by_overlapping_periods(data, overlap_starts, overlap_ends, if_average_overlap=False):
    """
    Finds the maximum value older data (already denormalized) in the overlapping period, takes the value from the
        same date from the newer data and calculates the scaling factor that the new data is multiplied by.

    Args:
        data:
        overlap_starts:
        overlap_ends:
        if_average_overlap:

    Returns:

    """
    pull_id = data.iloc[-1].name[0]
    normalized_data = data.loc[pull_id].iloc[:, 0].astype(np.float32)
    for overlap_start, overlap_end in zip(overlap_starts, overlap_ends):
        pull_id -= 1
        new_data = data.loc[pull_id].iloc[:, 0]
        normalized_overlap = normalized_data.loc[pd.Timestamp(overlap_start):pd.Timestamp(overlap_end)]
        max_normalized_overlap = normalized_overlap.max()
        max_normalized_overlap_id = normalized_overlap[normalized_overlap == max_normalized_overlap].index[0]
        new_data_reference_point = new_data.loc[max_normalized_overlap_id]
        scaling_factor = float(max_normalized_overlap) / new_data_reference_point
        new_data = new_data * scaling_factor
        if if_average_overlap is False:
            normalized_data_division_id = max_normalized_overlap_id + dt.timedelta(1)
            normalized_data = pd.concat(
                [new_data.loc[:max_normalized_overlap_id], normalized_data.loc[normalized_data_division_id:]], axis=0)
        else:
            averaged_overlap = average_overlap(new_data.loc[overlap_start:], normalized_data)
            # minus dt.timedelta(1) because loc includes the last index
            normalized_data = pd.concat(
                [new_data.loc[:overlap_start - dt.timedelta(1)], normalized_data], axis=0)
            normalized_data.loc[overlap_start:] = averaged_overlap.squeeze('columns')

        normalized_data = normalized_data / normalized_data.max() * 100.
    return normalized_data


def denormalize_by_overlapping_periods_maxes(data, overlap_starts, overlap_ends, if_average_overlap=False):
    """
    Finds maximum values of both data in the overlapping period and calculates a scaling factor that the new data will
        be multiplied by.
    Args:
        data:
        overlap_starts:
        overlap_ends:
        if_average_overlap: if False then the data will be create by putting the scaled data after the maximum point of
            the previously computed data

    Returns:

    """
    pull_id = data.iloc[-1].name[0]
    normalized_data = data.loc[pull_id].iloc[:, 0].astype(np.float32)
    for overlap_start, overlap_end in zip(overlap_starts, overlap_ends):
        pull_id -= 1
        new_data = data.loc[pull_id].iloc[:, 0]
        normalized_overlap = normalized_data.loc[pd.Timestamp(overlap_start):pd.Timestamp(overlap_end)]
        max_normalized_overlap = normalized_overlap.max()
        new_data_overlap = new_data.loc[pd.Timestamp(overlap_start):pd.Timestamp(overlap_end)]
        max_new_data_overlap = new_data_overlap.max()
        max_normalized_overlap_id = normalized_overlap[normalized_overlap == max_normalized_overlap].index[0]

        scaling_factor = float(max_normalized_overlap) / max_new_data_overlap
        new_data = new_data * scaling_factor
        if if_average_overlap is False:
            # loc includes lower and !upper! bound that's why the upper bound needs to one day later
            # note that it's better to increase the upper bound of the normalized data because there must be next index
            # if the new data index were decreased then if that was the last blog that and all blog would overlap then
            # a non-existing index would be chosen
            normalized_data_division_id = max_normalized_overlap_id + dt.timedelta(1)
            normalized_data = pd.concat(
                [new_data.loc[:max_normalized_overlap_id], normalized_data.loc[normalized_data_division_id:]], axis=0)
        else:
            averaged_overlap = average_overlap(new_data.loc[overlap_start:], normalized_data)
            # minus dt.timedelta(1) because loc includes the last index
            normalized_data = pd.concat(
                [new_data.loc[:overlap_start - dt.timedelta(1)], normalized_data], axis=0)
            normalized_data.loc[overlap_start:] = averaged_overlap.squeeze('columns')

        normalized_data = normalized_data / normalized_data.max() * 100.
    return normalized_data


def denormalize_daily_by_weekly(daily_data, weekly_data):
    if isinstance(daily_data.index, pd.MultiIndex):
        daily_data = daily_data.reset_index(0, drop=True).iloc[:, 0]
    if isinstance(weekly_data.index, pandas.MultiIndex):
        weekly_data = weekly_data.reset_index(0, drop=True).iloc[:, 0]
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
    kw_list = ["bitcoin"]
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(5 * 12 * 30 - 1)
    period = dt.timedelta(30 * 9 - 1)
    overlap_100 = dt.timedelta(100)
    overlap_0 = dt.timedelta(0)
    pytrends = TrendReq(hl='en-US', tz=360, retries=3, backoff_factor=0.1)
    pf_100 = PytrendsFetcher(pytrends)
    pf_0 = PytrendsFetcher(pytrends)
    pull_starts_100, pull_ends_100 = create_pulling_periods(start_date, end_date, overlap_100, period)
    pull_starts_0, pull_ends_0 = create_pulling_periods(start_date, end_date, overlap_0, period)
    result_100 = pull_overlapping_daily_data(pf_100, kw_list, pull_starts_100, pull_ends_100)
    result_0 = pull_overlapping_daily_data(pf_0, kw_list, pull_starts_0, pull_ends_0)
    overlap_starts, overlap_ends = create_overlap_periods(pull_starts_100, pull_ends_100, overlap_100)
    denormalized_by_overlapping_periods_max_with_avg = denormalize_by_overlapping_periods_maxes(result_100,
                                                                                                overlap_starts,
                                                                                                overlap_ends,
                                                                                                if_average_overlap=True)
