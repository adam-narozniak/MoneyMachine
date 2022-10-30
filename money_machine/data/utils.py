import datetime as dt


def str_date_to_date(str_date: str, date_format: str) -> dt.date:
    return dt.datetime.strptime(str_date, date_format)
