# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Aims

# The purpose of this notebook is to reproduce the paper and check how it applies to the present data.

# ## Imports

import pandas_datareader.data as web
import datetime as dt

# # Pick the companies

companies_to_tickers = {"Nike": "NKE", 
                       "Goldman Sachs": "GS", 
                       "Johnson and Johnson": "JNJ", 
                       "Pfizer": "PFE",
                       "JP Morgan Chase and Co": "JPM"}

tickers = list(companies_to_tickers.values())
tickers

# # Data

# ## Load

start_train = dt.datetime(2009, 4, 6)
end_train = dt.datetime(2017, 4, 3)
start_test = dt.datetime(2017, 4, 4)
end_test = dt.datetime(2019, 4, 5)

data_train = {ticker: web.DataReader(ticker, "yahoo", start_train, end_train)
             for ticker in tickers}
data_test = {ticker: web.DataReader(ticker, "yahoo", start_test, end_test)
             for ticker in tickers}

print(tickers[0])
data_train[tickers[0]].head()


# ## Add new variables

def add_new_vars(data):
    for ticker in tickers:
        data[ticker]["H-L"] = data[ticker]["High"] - data[ticker]["Low"]
        data[ticker]["O-C"] = data[ticker]["Open"] - data[ticker]["Close"]
        # Stock price’s seven days’ moving average (7 DAYS MA)
        # there is not explicitly said of what
        # so let's assume that it would be of the close price
        data[ticker]["MA-7d"] = data[ticker]["Close"].rolling(window=7).mean()
        data[ticker]["MA-14d"] = data[ticker]["Close"].rolling(window=14).mean()
        data[ticker]["MA-21d"] = data[ticker]["Close"].rolling(window=21).mean()
        data[ticker]["std-7d"] = data[ticker]["Close"].rolling(window=7).std()


add_new_vars(data_train)
add_new_vars(data_test)

# # Model

# ## Artificial Neural Network

# Since the detailed descritption is not provided about the training I'll infer the architecture from the model image and adjust the training time (epochs) and other hyperparams according to my preference.

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Input(shape=(7,)))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(optimizer="sgd", loss='mse')
