"""
Calibration of SVI(Search Volume Index) time series.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime as dt

from money_machine.data.gt.utils import determine_overlap, average_overlap


def calibrate_by_lin_reg(daily_from_live: pd.DataFrame, archive_data: pd.DataFrame):
    """
    Make live data resemble the historical data trend.
    The data is fit on the overlapping period.
    Args:
        daily_from_live: it's X, meaning we want it to change (assumed correct format)
        archive_data: it's y, meaning we aim to look like this (assumed correct format)

    Returns:
        Calibrated daily data coming created form live.

    """
    overlap_start, overlap_end = determine_overlap(daily_from_live, archive_data)
    X_common = daily_from_live.loc[overlap_start:overlap_end]  # both point s are included
    y_common = archive_data.loc[overlap_start:overlap_end]
    lin_reg = LinearRegression()
    lin_reg = lin_reg.fit(X_common.values.reshape(-1, 1), y_common.values.reshape(-1, 1))
    calibrated_daily_from_live = lin_reg.predict(daily_from_live.values.reshape(-1, 1))
    calibrated_daily_from_live = pd.DataFrame(calibrated_daily_from_live,
                                              index=daily_from_live.index,
                                              columns=daily_from_live.columns)
    return calibrated_daily_from_live


def calibrate_by_overlapping_periods(data,
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
                rather meaningless end data which is the most crucial)
            "chronological" - starts with start date and goes to the end date
            TODO:write something more

    TODO: add option to create the scaling point based on raw data on both sides
    Returns:

    """
    if order == "reverse-chronological":
        pull_id = data.iloc[-1].name[0]
    elif order == "chronological":
        pull_id = data.iloc[0].name[0]
    else:
        raise KeyError(f"There is no `order`: {order}."
                       f"Please refer to the documentation to find the available keywords.")
    calibrated_data = data.loc[pull_id].iloc[:, 0].astype(np.float32)
    for overlap_start, overlap_end in zip(overlap_starts, overlap_ends):
        if order == "reverse-chronological":
            pull_id -= 1
        elif order == "chronological":
            pull_id += 1
        new_data = data.loc[pull_id].iloc[:, 0]
        normalized_overlap = calibrated_data.loc[pd.Timestamp(overlap_start):pd.Timestamp(overlap_end)]
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
            calibrated_data_division_id = max_normalized_overlap_id + dt.timedelta(1)
            if order == "reverse-chronological":
                calibrated_data = pd.concat(
                    [new_data.loc[:max_normalized_overlap_id],
                     calibrated_data.loc[calibrated_data_division_id:]], axis=0)
            elif order == "chronological":
                calibrated_data = pd.concat(
                    [calibrated_data.loc[:max_normalized_overlap_id],
                     new_data.loc[calibrated_data_division_id:]], axis=0)
        else:
            averaged_overlap = average_overlap(new_data.loc[overlap_start:], calibrated_data)
            if order == "reverse-chronological":
                # minus dt.timedelta(1) because loc includes the last index
                calibrated_data = pd.concat(
                    [new_data.loc[:overlap_start - dt.timedelta(1)], calibrated_data], axis=0)
            elif order == "chronological":
                calibrated_data = pd.concat(
                    [calibrated_data, new_data.loc[overlap_end + dt.timedelta(1):]], axis=0)
            calibrated_data.loc[overlap_start:] = averaged_overlap.squeeze('columns')

    calibrated_data = calibrated_data / calibrated_data.max() * 100.
    return calibrated_data


def calibrate_daily_by_weekly(daily_data, weekly_data):
    if isinstance(daily_data.index, pd.MultiIndex):
        daily_data = daily_data.reset_index(0, drop=True).iloc[:, 0]
    if isinstance(weekly_data.index, pd.MultiIndex):
        weekly_data = weekly_data.reset_index(0, drop=True).iloc[:, 0]
    daily_calibrated_by_weekly = pd.concat([daily_data, weekly_data], axis=1).fillna(method="ffill")
    daily_calibrated_by_weekly.dropna(inplace=True)
    calibrated = daily_calibrated_by_weekly.iloc[:, 0] * daily_calibrated_by_weekly.iloc[:, 1]
    calibrated = calibrated / calibrated.max() * 100
    return calibrated


def denormalize_daily_with_overlapping_periods_by_weekly(data, weekly_data):
    pull_ids = np.unique(data.index.get_level_values(0))
    calibrated = pd.DataFrame()
    for pull_id in pull_ids:
        temp_data = data.loc[pull_id].iloc[:, 0]
        adj = calibrate_daily_by_weekly(temp_data, weekly_data)
        adj = adj.to_frame()
        adj["pull_id"] = pull_id
        adj = adj.set_index(["pull_id", adj.index])
        calibrated = pd.concat([calibrated, adj], axis=0)
    return calibrated
