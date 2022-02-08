"""
Module to fetch for Google Trends data and perform appropriate scaling.

Conventions:
    Google Trends Data is kept in pd.DataFrame objects where the index is dt.datetime.
    If it is concatenated together for many different periods it has MultiIndex: "pull_id", "date".
"""
import datetime as dt
import pathlib
import time
from typing import Union

import numpy as np
import pandas
import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm

from money_machine.data.pytrends_fetcher import PytrendsFetcher

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


def create_live_pulling_period(data_for_day: dt.date):
    """
    data for day: created by this function data will be used to predict that day
    """
    # data_for_day_datetime: it doesnt' mean that the prediction will be made for this time, it's just a convince
    data_for_day_datetime = dt.datetime.combine(data_for_day, dt.time(0))
    live_period_start = data_for_day_datetime - dt.timedelta(7)
    live_period_end = data_for_day_datetime - dt.timedelta(hours=1)
    return live_period_start, live_period_end


def create_live_pulling_periods(data_for_days: list[dt.date]):
    starts = []
    ends = []
    for d in data_for_days:
        start, end = create_live_pulling_period(d)
        starts.append(start)
        ends.append(end)
    return starts, ends


def transform_hourly_to_daily(hourly_data):
    """Averages weekly hourly data into daily."""
    hourly_data = hourly_data.iloc[:, 0]
    dates = pd.to_datetime(hourly_data.index.get_level_values(1).values).date
    hourly_data = hourly_data.to_frame().set_index(dates, append=True)
    hourly_data.index = hourly_data.index.set_names("day", level=2)
    daily_data = hourly_data.groupby(level=[0, 2]).mean()
    return daily_data


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
                           period: dt.timedelta,
                           chronological_order: bool = True,
                           assert_repeats: bool = False) -> tuple[list[dt.date], list[dt.date]]:
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
        chronological_order: whether to start creating from start_date or end_date; the result is that
            the last (in chronological_order) or the first period might be shorter
        assert_repeats: if True then the there will be at most 2 datapoint for the same date, otherwise might be more
    Note:
        The last period might have equal or smaller overlapping period. It's because the there are probably not discrete
         number of parts of periods with overlaps in the specified timeframe. The normal procedure is to decrease the
         last period.
    Returns:
        chronologically ordered lists of starts and ends
    """
    delta = overlap - dt.timedelta(1)
    input_period = period
    period = period - dt.timedelta(1)  # this is due to the fact that we count the end day as a whole
    period_minus_two_deltas = period - 2 * delta
    if assert_repeats:
        if period_minus_two_deltas < dt.timedelta(0):
            # minus ones here are for the human interpretability for words overlap and period
            raise Exception(f"The period should be greater or equal that 2*delta where delta = (overlap -1).\n"
                            f"The values provided are period: {input_period}, overlap: {overlap}.\n"
                            f"They  don't satisfy the condition.\n"
                            f"For given period: {input_period}, the biggest overlap which you can give as an argument "
                            f"is {period / 2 + dt.timedelta(1)}")
    date_diff = end_date - start_date
    check_fetch_data_correctness(start_date, end_date, delta, date_diff)
    starts = []
    ends = []
    if chronological_order:
        current_start_date = start_date
        current_end_date = start_date + period
        while current_end_date < end_date + dt.timedelta(1):
            starts.append(current_start_date)
            ends.append(current_end_date)
            current_start_date = current_start_date + period - delta
            current_end_date = current_end_date + period - delta
        # add the last period only if it's needed, meaning when the last computed end date is not the same as end_date
        if current_end_date - period + delta != end_date:
            # ends[-1] - delta is the same as the normal start would be at that time
            # it could be also left as current_start_date
            current_start_date = ends[-1] - delta
            current_end_date = end_date
            starts.append(current_start_date)
            ends.append(current_end_date)
    else:
        current_end_date = end_date
        current_start_date = end_date - period
        while current_start_date > start_date - dt.timedelta(1):
            starts.append(current_start_date)
            ends.append(current_end_date)
            current_start_date = current_start_date - period + delta
            current_end_date = current_end_date - period + delta
        if current_start_date + period - delta != start_date:
            current_start_date = start_date
            # current_end_date might have been left unchanged this line doesn't change it's value
            current_end_date = starts[-1] + delta
            starts.append(current_start_date)
            ends.append(current_end_date)
        starts.reverse()
        ends.reverse()
    assert (len(starts) == len(ends))
    return starts, ends


def pull_data(fetcher,
              kw_list: list[str],
              start_dates: list[dt.date],
              end_dates: list[dt.date],
              timeframe_type: str = "date"):
    """
    Fetches the data from the google trends using interface from Fetcher.
    Args:
        fetcher: an instance that inherits from Fetcher
        kw_list: search keywords list
        start_dates:
        end_dates:
        timeframe_type: either date or datetime

    Returns:
        data for given periods concatenated together.

    TODO: so check that the timedelta is the same for every entry in each pull
    """
    cache_dir = pathlib.Path(f"/Users/adamnarozniak/Projects/MoneyMachine/cache/")
    if_empty = not any(cache_dir.iterdir())
    if not if_empty:
        files = cache_dir.glob("*")
        last_cache = max(files, key=lambda x: x.stat().st_ctime)
        result = pd.read_pickle(last_cache)
        last_read_point = result.index.get_level_values(0).max()
    else:
        result = pd.DataFrame()
        last_read_point = -1
    cache_path = cache_dir / pathlib.Path(f"data_{kw_list[0]}_{time.time_ns()}.pkl")
    n_pulls = len(start_dates)
    missed_starts = []
    missed_ends = []
    missed_pull_ids = []
    for pull_id, (current_start_date, current_end_date) in enumerate(tqdm(zip(start_dates, end_dates), total=n_pulls)):
        if pull_id <= last_read_point:
            continue
        if timeframe_type == "date":
            timeframe = create_timeframe_from_datetime(current_start_date, current_end_date)
        elif timeframe_type == "datetime":
            timeframe = create_hourly_timeframe(current_start_date, current_end_date)
        else:
            raise KeyError(f"Only 'date' and 'datetime' timeframes are supported. "
                           f"You provided {timeframe_type} instead")
        new_data = fetcher.fetch_data(kw_list, timeframe)
        if new_data.empty:
            missed_starts.append(current_start_date)
            missed_ends.append(current_end_date)
            missed_pull_ids.append(pull_id)
        new_data["pull_id"] = pull_id
        new_data.set_index(["pull_id", new_data.index], inplace=True)
        result = pd.concat([result, new_data], axis=0)
        result.to_pickle(str(cache_path))
    if len(missed_starts) != 0:
        print(f"There are {len(missed_starts)} results. Recursive call.")
        missed_result = pull_data(fetcher, kw_list, missed_starts, missed_ends, timeframe_type)
        missed_result = missed_result.reset_index()
        missed_result["pull_id"] = missed_pull_ids
        missed_result = missed_result.set_index(["pull_id", "day"])
        result = pd.concat([result, missed_result], axis=0)
    return result


def pull_overlapping_daily_data(fetcher, kw_list: list[str], start_dates: list[dt.date], end_dates: list[dt.date]):
    return pull_data(fetcher, kw_list, start_dates, end_dates, timeframe_type="date")


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
        assert dates_diff == overlap, f"The date {b} should be later than {a} by exactly {overlap}, " \
                                      f"but is {dates_diff} instead"
    return pull_starts, pull_ends


def denormalize_by_overlapping_periods(data,
                                       overlap_starts,
                                       overlap_ends,
                                       strategy="max-to-max",
                                       if_average_overlap=False,
                                       order="reverse-chronological"):
    """
    Denormalized data based on scaling factor calculated based on the overlapping periods of two data based on the
        specified `strategy` and `if_average_overlap`.
    Args:
        data:
        overlap_starts:
        overlap_ends:
        strategy: "max-to-point" or "max-to-max", method to used when creating scaling factor
            "max-to-max" - find the max of both data in the overlapping period, (WARNING: it might be a mistake to take
                the max from the already calibrated data and a new data that is not calibrated)
            "max-to-point" - find the max of the already calibrated (old) data from the overlapping period and uses that
                point with the new data value coming form the same date (WARNING: it might be a mistake to take it from
                already calibrated data instead from the raw data)
        if_average_overlap: if False then the data will be created by putting the scaled data after the maximum point of
            the previously computed data
        order: order in which the scaling will be performed, note the result will also depend on the overlap periods
            you provide
            "reverse-chronological" - starting from the end date and going to the start (drawback is that you have
                rather meaningless end data which is the most crutial)
            "chronological" - starts with start date and goes to the end date
            TODO:write something more

    TODO: add option to create the scaling point based on raw data on both sides
    Returns:

    """
    original_data = data.copy()
    if order == "reverse-chronological":
        pull_id = data.iloc[-1].name[0]
    elif order == "chronological":
        pull_id = data.iloc[0].name[0]
    else:
        raise KeyError(f"There is no `order`: {order}."
                       f"Please refer to the documentation to find the available keywords.")
    normalized_data = data.loc[pull_id].iloc[:, 0].astype(np.float32)
    for overlap_start, overlap_end in zip(overlap_starts, overlap_ends):
        if order == "reverse-chronological":
            pull_id -= 1
        elif order == "chronological":
            pull_id += 1
        new_data = data.loc[pull_id].iloc[:, 0]
        normalized_overlap = normalized_data.loc[pd.Timestamp(overlap_start):pd.Timestamp(overlap_end)]
        max_normalized_overlap = normalized_overlap.max()
        max_normalized_overlap_id = normalized_overlap[normalized_overlap == max_normalized_overlap].index[0]
        if strategy == "max-to-point":
            # get the date of the maximum value of the already calibrated data
            new_data_reference_point = new_data.loc[max_normalized_overlap_id]
        elif strategy == "max-to-max":
            new_data_overlap = new_data.loc[pd.Timestamp(overlap_start):pd.Timestamp(overlap_end)]
            new_data_reference_point = new_data_overlap.max()
        else:
            raise KeyError(f"There is no strategy: {strategy}."
                           f"Please refer to the documentation to find the available keywords.")
        scaling_factor = float(max_normalized_overlap) / new_data_reference_point
        new_data = new_data * scaling_factor
        if if_average_overlap is False:
            # loc includes lower and !upper! bound that's why the upper bound needs to one day later
            # note that it's better to increase the upper bound of the normalized data because there must be next index
            # if the new data index were decreased then if that was the last blog that and all blog would overlap then
            # a non-existing index would be chosen
            normalized_data_division_id = max_normalized_overlap_id + dt.timedelta(1)
            if order == "reverse-chronological":
                normalized_data = pd.concat(
                    [new_data.loc[:max_normalized_overlap_id],
                     normalized_data.loc[normalized_data_division_id:]], axis=0)
            elif order == "chronological":
                normalized_data = pd.concat(
                    [normalized_data.loc[:max_normalized_overlap_id],
                     new_data.loc[normalized_data_division_id:]], axis=0)
        else:
            averaged_overlap = average_overlap(new_data.loc[overlap_start:], normalized_data)
            if order == "reverse-chronological":
                # minus dt.timedelta(1) because loc includes the last index
                normalized_data = pd.concat(
                    [new_data.loc[:overlap_start - dt.timedelta(1)], normalized_data], axis=0)
            elif order == "chronological":
                normalized_data = pd.concat(
                    [normalized_data, new_data.loc[overlap_end + dt.timedelta(1):]], axis=0)
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
    denormalized_by_overlapping_periods_max = denormalize_by_overlapping_periods(result_100,
                                                                                 overlap_starts,
                                                                                 overlap_ends,
                                                                                 strategy="max-to-max",
                                                                                 if_average_overlap=False)
    denormalized_by_overlapping_periods_max_chrono = denormalize_by_overlapping_periods(result_100,
                                                                                        overlap_starts[::-1],
                                                                                        overlap_ends[::-1],
                                                                                        strategy="max-to-max",
                                                                                        if_average_overlap=False,
                                                                                        order="chronological")
