import numpy as np
import pandas as pd


def check_args(fnc, n):
    fnc_name = fnc.__name__
    if n <= 0:
        raise ValueError(f"n parameter in function {fnc_name} has to be positive but instead n = {n} was given")


def lowest_close_n(data, n):
    n_min = data["Close"].rolling(window=n).apply(lambda x: np.min(x))
    return n_min


def highest_close_n(data, n):
    n_max = data["Close"].rolling(window=n).apply(lambda x: np.max(x))
    return n_max
