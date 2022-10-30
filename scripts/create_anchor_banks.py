import datetime as dt
import pathlib

import gtab

from money_machine.data.gt.gt_data import create_pulling_periods, create_anchor_banks
from money_machine.data.gt.gtab_fetcher import GtabFetcher

if __name__ == "__main__":
    end_date = dt.date(2022, 2, 1)
    start_date = end_date - dt.timedelta(5 * 12 * 30 - 1)
    period = dt.timedelta(30 * 9 - 1)  # this is the maximum daily period
    overlap = dt.timedelta(100)
    start_dates, end_dates = create_pulling_periods(start_date, end_date, overlap, period)
    setup_path = pathlib.Path("./data/gtab/test_20220204")
    t = gtab.GTAB(str(setup_path))
    gtab_fetcher = GtabFetcher(t, str(setup_path))
    create_anchor_banks(gtab_fetcher, start_dates, end_dates)
