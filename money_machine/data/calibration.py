"""
Calibration of SVI(Search Volume Index) time series.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression

from money_machine.data.gt_data import determine_overlap


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
    calibrated_daily_from_live.index = pd.to_datetime(calibrated_daily_from_live.index)
    return calibrated_daily_from_live
