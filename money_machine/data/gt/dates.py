from typing import Union
import datetime as dt

import pandas as pd

from money_machine.data.gt.utils import check_fetch_data_correctness


def create_live_pulling_period_based_on_prediction_dates(data_for_day: dt.date):
    """
    data for day: created by this function data will be used to predict that day
    """
    # data_for_day_datetime: it doesnt' mean that the prediction will be made for this time, it's just a convince
    data_for_day_datetime = dt.datetime.combine(data_for_day, dt.time(0))
    live_period_start = data_for_day_datetime - dt.timedelta(7)
    live_period_end = data_for_day_datetime - dt.timedelta(hours=1)
    return live_period_start, live_period_end


def create_live_pulling_periods_based_on_prediction_dates(
        prediction_dates: Union[list[dt.date], tuple[dt.date, dt.date]]):
    """
    Gives pulling periods for archive data (NON-real time data)
    Args:
        prediction_dates: list of days that you need prediction for or first and last date of prediction

    Returns:

    """
    starts = []
    ends = []
    for d in prediction_dates:
        start, end = create_live_pulling_period_based_on_prediction_dates(d)
        starts.append(start)
        ends.append(end)
    return starts, ends


def create_prediction_dates(first_prediction_date: dt.date, last_prediction_date: dt.date):
    """
    Based on the first and last day that you neeed to make prediction for the function returns all
    dates between (both included)
    Args:
        last_prediction_date:
        first_prediction_date :

    Returns:

    """
    return pd.date_range(first_prediction_date, last_prediction_date)


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