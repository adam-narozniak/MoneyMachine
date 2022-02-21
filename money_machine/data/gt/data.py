"""
Module to fetch for Google Trends data and perform appropriate scaling.

Conventions:
    Google Trends Data is kept in pd.DataFrame objects where the index is dt.datetime.
    If it is concatenated together for many different periods it has MultiIndex: "pull_id", "date".
"""
import datetime as dt
import pathlib
import time

import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm

from money_machine.data.gt.calibration import calibrate_by_lin_reg, calibrate_by_overlapping_periods
from money_machine.data.gt.dates import create_pulling_periods, create_overlap_periods
from money_machine.data.gt.pytrends_fetcher import PytrendsFetcher
from money_machine.data.gt.utils import create_timeframe_from_datetime, create_hourly_timeframe, \
    transform_hourly_to_daily, merge_data


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
    cache_dir = pathlib.Path(f"/cache/")
    if_empty = not any(cache_dir.iterdir())
    if not if_empty:
        files = cache_dir.glob("*")
        last_cache = max(files, key=lambda x: x.stat().st_ctime)
        result = pd.read_pickle(last_cache)
        read_pulls = set(result.index.get_level_values(0).unique().to_list())
    else:
        result = pd.DataFrame()
        read_pulls = set([])
    cache_path = cache_dir / pathlib.Path(f"data_{kw_list[0]}_{time.time_ns()}.pkl")
    n_pulls = len(start_dates)
    missed_starts = []
    missed_ends = []
    missed_pull_ids = []
    for pull_id, (current_start_date, current_end_date) in enumerate(tqdm(zip(start_dates, end_dates), total=n_pulls)):
        if pull_id in read_pulls:
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
            dates_diff = (current_end_date - current_start_date).days * 24
            missed_starts.append(current_start_date)
            missed_ends.append(current_end_date)
            missed_pull_ids.append(pull_id * dates_diff)
        new_data["pull_id"] = pull_id
        new_data.set_index(["pull_id", new_data.index], inplace=True)
        result = pd.concat([result, new_data], axis=0)
        result.to_pickle(str(cache_path))
    # that is not a reliable solution; GOOGLE data seems to have leaks for that period
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


def find_faulty_pull_ids(data, strategy="strict"):
    """
    {"strict"|"last"}
    Two types of faulty pull ids are returned
    1) the missing ids (google trends doesn't have data for the whole periods)
    2) partially missing data = zeros (heuristic applied here is to treat the data from a pull as partially missing
        the last value must be zero)"""
    missing_ids = []
    incomplete_ids = []
    ids = data.index.get_level_values(0).unique()
    max_id = ids.max()
    for i in list(range(max_id + 1)):
        try:
            pull_data = data.loc[i]
            if strategy == "strict":
                if (pull_data.iloc[:, 0].values == 0).any():
                    incomplete_ids.append(i)
            elif strategy == "last":
                if pull_data.iloc[-1].values[0] == 0:
                    incomplete_ids.append(i)
        except KeyError:
            missing_ids.append(i)
    return missing_ids, incomplete_ids


def get_full_substitute_of_live(archive_data, current_pull_id, archive_to_live_delta):
    # so if I am missing a value for my last day period I need to pull an archive for
    # the number of delta days
    pulls_diff = archive_to_live_delta.days
    forward_archive = archive_data.loc[current_pull_id + pulls_diff].iloc[:, 0].to_frame()
    end_live_date = forward_archive.iloc[-1].name
    start_live_date = end_live_date - dt.timedelta(6)  # 6 not 7 because we want 7-day period
    daily_from_live = forward_archive.loc[start_live_date:end_live_date]
    return daily_from_live


def create_real_time_dataset(live_data, archive_data, missing_ids, incomplete_ids, fit_methods="lin_reg"):
    prediction_day_to_dataset = {}
    ids = live_data.index.get_level_values(0).unique()
    max_id = ids.max()
    faulty_ids = sorted(missing_ids + incomplete_ids)
    cumulated_archive = archive_data.loc[0].iloc[:, 0].to_frame()
    # missing_pull_ids = find_missing_pull_ids(live_data)
    for i in ids:

        # transform live to daily (differently based on the type)
        if i in missing_ids:
            # the live data creation will have to be fully recreated
            archive_to_live_delta = dt.timedelta(3)  # Though I think it's 2 ; TODO: check it well
            daily_from_live = get_full_substitute_of_live(archive_data, i, archive_to_live_delta)
        elif i in incomplete_ids:
            current_live = live_data.loc[i]
            # fix the icomplete periods?
            # the days in which the incomplete period exist will be
            # changed to the days from archive data
            # thought the partially fine data has to be fit to the archive data
            # which archive data to choose?
            # the one that violates the reality to the samllest degree

            # I can also apply here few heuristics
            # 1. fully archive data
            # 2. lin reg fit if the earlier period is at least half full and there exist
            # the data for the prediction days (i'm not sure about that)
            # 3. fit all you have take the rest from archive
            # strategy 1
            archive_to_live_delta = dt.timedelta(3)  # Though I think it's 2 ; TODO: check it well
            daily_from_live = get_full_substitute_of_live(archive_data, i, archive_to_live_delta)
            daily_from_live.index = pd.to_datetime(daily_from_live.index)

        else:
            current_live = live_data.loc[i]
            daily_from_live = transform_hourly_to_daily(current_live, amount="single")

        # fit daily_from_live to daily
        # current_archive = archive_data.loc[i].iloc[:, 0].to_frame() # old way of doing that, now pull for that and have it adjusted by older stuff
        current_archive = archive_data.loc[i].iloc[:, 0].to_frame()
        adjusted_current_archive = calibrate_by_lin_reg(current_archive, cumulated_archive)
        cumulated_archive = merge_data(cumulated_archive, adjusted_current_archive, strategy="average")

        adjusted_daily = calibrate_by_lin_reg(daily_from_live, cumulated_archive)

        merged = merge_data(cumulated_archive, adjusted_daily, strategy="average")
        prediction_day = merged.index.date[-1] + dt.timedelta(1)
        # that's the data you have for the prediction for the next day
        prediction_day_to_dataset[prediction_day] = merged
    return prediction_day_to_dataset


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
    denormalized_by_overlapping_periods_max = calibrate_by_overlapping_periods(result_100,
                                                                               overlap_starts,
                                                                               overlap_ends,
                                                                               strategy="max-to-max",
                                                                               if_average_overlap=False)
    denormalized_by_overlapping_periods_max_chrono = calibrate_by_overlapping_periods(result_100,
                                                                                      overlap_starts[::-1],
                                                                                      overlap_ends[::-1],
                                                                                      strategy="max-to-max",
                                                                                      if_average_overlap=False,
                                                                                      order="chronological")
