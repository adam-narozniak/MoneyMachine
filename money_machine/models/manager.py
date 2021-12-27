"""Module to perform common training operations."""
from money_machine.models.utils import load_config
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from money_machine.evaluation.metrics import calculate_all_metrics
from money_machine.visualization.stock import plot_stock


class Manager:
    def __init__(self, model_, train_config_path, scaler_X=None, scaler_y=None):
        self.model = model_
        self.train_config = load_config(train_config_path)
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.X_train = None
        self.y_train = None
        self.y_train_pd = None

        self.X_train_sc = None
        self.y_train_sc = None

        self.X_test = None
        self.y_test = None
        self.y_test_pd = None

        self.X_test_sc = None
        self.y_test_sc = None

        self.y_pred = None
        self.y_pred_sc = None

        self.evaluation = {"mae": None, "mape": None, "mse": None}

    def set_train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train_pd = y_train
        self.y_train = y_train.values

    def set_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test_pd = y_test
        self.y_test = y_test.values

    def scale_X_y(self, scale_y=True):
        if self.scaler_X is None:
            self.scaler_X = MinMaxScaler()
        if self.scaler_y is None:
            self.scaler_y = MinMaxScaler()
        self.X_train_sc = self.scaler_X.fit_transform(self.X_train)
        self.X_test_sc = self.scaler_X.transform(self.X_test)
        if scale_y:
            self.y_train_sc = self.scaler_y.fit_transform(self.y_train)
            self.y_test_sc = self.scaler_y.transform(self.y_test)

    def expand_dim_to_3(self):
        if self.X_train.ndim == 2:
            self.X_train_sc = np.expand_dims(self.X_train_sc, 1)
        if self.X_test.ndim == 2:
            self.X_test_sc = np.expand_dims(self.X_test_sc, 1)

    def compile(self):
        self.model.compile(loss=self.train_config["loss"], optimizer=self.train_config["optimizer"])

    def train(self):
        self.model.fit(self.X_train_sc, self.y_train_sc, epochs=self.train_config["epochs"], shuffle=False)

    def print_summary(self):
        print(self.model.summary())

    def predict(self):
        self.y_pred_sc = self.model.predict(self.X_test_sc)
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
        plot_stock(self.y_test, self.y_pred, self.y_test_pd.index.values, n_ahead)
