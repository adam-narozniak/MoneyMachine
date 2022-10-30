import pathlib
from typing import Union

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from money_machine.data.data import *
from money_machine.evaluation.metrics import calculate_all_metrics
from money_machine.models.utils import load_config
from money_machine.visualization.stock import plot_stock


class Manager:
    def __init__(self, model: tf.keras.Model, train_config_path: pathlib.Path):
        self.model = model
        self.train_config = load_config(train_config_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.scaler_X = None
        self.scaler_y = None

        self.X_train_sc = None
        self.y_train_sc = None
        self.X_test_sc = None
        self.y_test_sc = None

        self.y_pred = None
        self.y_pred_sc = None

        self.division = None

        self.walk_forward_hist = []

        self.evaluation = {"mae": None, "mape": None, "mse": None}

    def set_X_y(self, X, y):
        self.X = X
        self.y = y

    def set_same_date_X_y(self, X, y, n_days_ahead):
        y = create_y_for_n_days_ahead(y, n_days_ahead)
        common_index = find_common_index(X.index, y.index)
        self.y = y.loc[common_index]
        self.X = X.loc[common_index]

    def divide_train_test(self, division: Union[float, dt.datetime]):
        """

        Args:
            division: when int the % of the train data, when dt.datetime the last date of the train

        Returns:

        """
        self.division = division
        if isinstance(division, dt.date):
            self.X_train, self.X_test = self.X.loc[:division], self.X.loc[division + dt.timedelta(1):]
            self.y_train, self.y_test = self.y.loc[:division], self.y.loc[division + dt.timedelta(1):]
        else:
            raise NotImplementedError("To implement")

    def scale(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.X_train_sc = pd.DataFrame(self.scaler_X.fit_transform(self.X_train), index=self.X_train.index,
                                       columns=self.X_train.columns)
        self.X_test_sc = pd.DataFrame(self.scaler_X.transform(self.X_test), index=self.X_test.index,
                                      columns=self.X_test.columns)

        self.y_train_sc = pd.DataFrame(self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)),
                                       index=self.y_train.index)
        self.y_test_sc = pd.DataFrame(self.scaler_y.transform(self.y_test.values.reshape(-1, 1)),
                                      index=self.y_test.index)

    def make_multistep(self, n_previous_days):
        X = pd.concat([self.X_train_sc, self.X_test_sc], axis=0)
        y = pd.concat([self.y_train_sc, self.y_test_sc], axis=0)
        common_index = find_common_index(X.index, y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        X = generate_multitimestep_data(X, n_previous_days)
        X = X.dropna(axis=0)
        common_index = find_common_index(X.index, y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        beg_date, end_date = common_index[0], common_index[-1]
        self.X_train_sc = X.loc[beg_date: self.division]
        self.X_test_sc = X.loc[self.division + dt.timedelta(1): end_date]
        self.y_train_sc = y.loc[beg_date: self.division]
        self.y_test_sc = y.loc[self.division + dt.timedelta(1): end_date]

    def expand_to_3_dims(self):
        """
        It assumes that the number of fetures is one and that the multitimestep is used.
        """
        self.X_train_sc = self.X_train_sc.values.reshape(self.X_train_sc.shape[0], self.X_train_sc.shape[1], -1)
        self.X_test_sc = self.X_test_sc.values.reshape(self.X_test_sc.shape[0], self.X_test_sc.shape[1], -1)

    def setup(self, raw_X, raw_y, division, n_days_ahead, n_previous_days):
        """
        It assumes that the X and y has the same dates.
        Args:
            raw_X:
            raw_y:
            division:
            n_days_ahead:
            n_previous_days:

        Returns:

        """
        self.set_same_date_X_y(raw_X, raw_y, n_days_ahead)
        self.divide_train_test(division)
        self.scale()
        self.make_multistep(n_previous_days)
        self.expand_to_3_dims()

    def compile(self):
        self.model.compile(loss=self.train_config["loss"], optimizer=self.train_config["optimizer"])

    def train(self):
        return self.model.fit(self.X_train_sc, self.y_train_sc, epochs=self.train_config["epochs"], shuffle=False)

    def print_summary(self):
        print(self.model.summary())

    def predict(self):
        self.y_pred_sc = self.model.predict(self.X_test_sc)
        self.y_pred = self.scaler_y.inverse_transform(self.y_pred_sc)

    def predict_walk_forward(self, train_time=1):
        """
        After every single prediction the model is trained on old X_train plus an additional data point from the
            test dataset.

        Args:
            train_time: number of epochs to train every single new instance

        """
        y_pred_scs = []
        X_train_plus_one = self.X_train_sc.copy()
        y_train_plus_one = self.y_train_sc.copy()
        test_len = self.y_test_sc.shape[0]
        for x_test, y_test in tqdm(zip(self.X_test_sc, self.y_test_sc.values), total=test_len):
            x_test = np.expand_dims(x_test, 0)
            y_test = np.expand_dims(y_test, 1)
            y_pred_sc = self.model.predict(x_test)
            y_pred_scs.append(y_pred_sc)
            X_train_plus_one = np.concatenate([X_train_plus_one, x_test])
            y_train_plus_one = np.concatenate([y_train_plus_one, y_test])
            hist = self.model.fit(X_train_plus_one, y_train_plus_one, epochs=train_time, shuffle=False, verbose=0)
            self.walk_forward_hist.append(hist)
        self.y_pred_sc = np.array(y_pred_scs).reshape(-1, 1)
        self.y_pred = self.scaler_y.inverse_transform(self.y_pred_sc)

    def evaluate(self):
        mae, mape, mse = calculate_all_metrics(self.y_test, self.y_pred)
        self.evaluation["mae"] = mae
        self.evaluation["mape"] = mape
        self.evaluation["mse"] = mse

    def print_eval(self):
        for key, val in self.evaluation.items():
            print(f"{key}: {val}")

    def visualize(self, n_ahead):
        plot_stock(self.y_test, self.y_pred, self.X_test.index.values, n_ahead)
