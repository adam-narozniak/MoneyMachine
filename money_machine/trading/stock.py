import numpy as np


class Stock:
    def __init__(self, pandas_prices):
        self.pandas_prices = pandas_prices
        self.prices = pandas_prices.values
        self.dates = pandas_prices.index.values

    def get_prices_for(self, date):
        return self.pandas_prices.loc[date][0]

    def get_prices_for_next(self, date):
        idx = np.searchsorted(self.pandas_prices.index, date)
        return self.prices[idx + 1]