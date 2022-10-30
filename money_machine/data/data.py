import datetime as dt

import pandas as pd


def load_saved_archive_data(path):
    """Loads data with index: 'pull_id', 'date'."""
    data = pd.read_csv(path, index_col=0)
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date", append=True)
    return data


def generate_multitimestep_data(data: pd.DataFrame, n_additional_days: int) -> pd.DataFrame:
    """
    Gather n_additional_days + 1 data points as a single instance.

    It accomplishes that by shifting the whole data and concatenating it together.
    Note that first few rows will have nan (due to the lack of the data).

    Args:
        data:
        n_additional_days: number of additional days to have in a single row

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


def drop_nans(data):
    return data.dropna(axis=0)


def divide_test_train(data, date):
    data_train, data_test = data.loc[:pd.Timestamp(date)], data.loc[pd.Timestamp(date) + dt.timedelta(1):]
    return data_train, data_test


def create_y_for_n_days_ahead(y, n):
    """
    Shifts the data so the prediction for n days ahead is possible.
    Args:
        y: y data
        n: number of days to shift

    Returns:
        shifted y data
    """
    y = y.shift(-n)
    y.name = "y"
    return y


def find_common_beg_end_date(index1, index2):
    """
    Returns the beginning and end date for the two indexes so that they both have the give period.
    Args:
        index2:
        index1:

    Returns:
        beginning date, end date
    """
    beg_date = max(index1[0], index2[0])
    end_date = min(index1[-1], index2[-1])
    return beg_date, end_date


def find_common_index(index1, index2):
    return index1.intersection(index2)
