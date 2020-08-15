import tensorflow.keras.backend as K
import tensorflow as tf 
from tensorflow.keras.layers import Layer
import numpy as np

class GatherThePoint(Layer):
    def __init__(self, batch_sample_xyz, dtype=tf.float32):
        self.xyz = batch_sample_xyz
        super(GatherThePoint, self).__init__()

    def call(self, input):
        result = tf.map_fn(lambda param:tf.gather_nd(param[0],tf.expand_dims(param[1],-1)),elems=(self.xyz,input),dtype=self.dtype)
        return result
