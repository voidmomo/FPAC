import tensorflow as tf 
from tensorflow.keras.layers import Layer

class Mish(Layer):
    def __init__(self):
        super(Mish, self).__init__()
    def call(self, input):
        return input * tf.math.tanh(tf.nn.softplus(input))