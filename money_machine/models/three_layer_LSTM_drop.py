import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout

from money_machine.models.utils import load_config


class ThreeLayerLSTM(tf.keras.Model):
    def __init__(self, config_path):
        super(ThreeLayerLSTM, self).__init__()
        self.config = load_config(config_path)
        self.input_block = Input(shape=(self.config["input_shape_0"], self.config["input_shape_1"]))
        self.lstm_1 = LSTM(self.config["neurons"], stateful=False, return_sequences=True)
        self.drop_1 = Dropout(self.config["dropout"])
        self.lstm_2 = LSTM(self.config["neurons"], stateful=False, return_sequences=True)
        self.drop_2 = Dropout(self.config["dropout"])
        self.lstm_3 = LSTM(self.config["neurons"], stateful=False)
        self.drop_3 = Dropout(self.config["dropout"])
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.drop_1(x)
        x = self.lstm_2(x)
        x = self.drop_2(x)
        x = self.lstm_3(x)
        x = self.drop_3(x)
        x = self.dense(x)
        return x
