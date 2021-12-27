import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM

from money_machine.models.utils import load_config


class SingleLayerLSTM(tf.keras.Model):
    def __init__(self, config_path):
        super(SingleLayerLSTM, self).__init__()
        self.config = load_config(config_path)
        self.input_block = Input(shape=(self.config["input_shape_0"], self.config["input_shape_1"]))
        self.lstm = LSTM(self.config["neurons"], stateful=False)
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x
