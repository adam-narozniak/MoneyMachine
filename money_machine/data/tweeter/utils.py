import re


def parse_date_to_hashtag_count_list(date_to_hashtag_count_list):
    clean = re.sub("[\[\],\s]", "", date_to_hashtag_count_list)
    splitted = re.split("[\'\"]", clean)
    values_only = [s for s in splitted if s != '']
    return values_only
