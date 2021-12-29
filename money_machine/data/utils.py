import datetime as dt
import re


def _parse_date_to_hashtag_count_list(date_to_hashtag_count_list):
    clean = re.sub("[\[\],\s]", "", date_to_hashtag_count_list)
    splitted = re.split("[\'\"]", clean)
    values_only = [s for s in splitted if s != '']
    return values_only


def _str_dates_to_dates(str_dates):
    date_format = "%Y/%m/%d"
    return [dt.datetime.strptime(str_date, date_format) for str_date in str_dates]
