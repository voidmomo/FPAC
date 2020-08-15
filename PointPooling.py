import tensorflow.keras.backend as K
import tensorflow as tf 
from tensorflow.keras.layers import Layer
import numpy as np

class PointPooling(Layer):
    def __init__(self, batch_sample_xyz, sampling, poolN, dtype=tf.float32):
        
        self.xyz = batch_sample_xyz
        self.sampling = sampling
        self.poolN = poolN
        super(PointPooling, self).__init__()

    def call(self, input):
        new_adj = tf.expand_dims(self.sampling, axis=1)
        new_adj = tf.subtract(new_adj, tf.transpose(tf.expand_dims(self.xyz, axis=1), [0, 2, 1, 3]))
        new_adj = tf.square(new_adj)
        new_adj = tf.reduce_sum(new_adj, axis=3) #按照坐标的一轴进行相加.

        _,idx = tf.nn.top_k(-tf.transpose(new_adj,[0,2,1]),self.poolN)
        idx = tf.expand_dims(idx,3)
        #xyz_slices = tf.map_fn(lambda param:tf.gather_nd(param[0],param[1]),elems=(self.xyz,idx),dtype=self.dtype)  #B,N,maxn，3
        further_slices = tf.map_fn(lambda param:tf.gather_nd(param[0],param[1]),elems=(input,idx),dtype=self.dtype)  #B,NEWSAMPLING_N,9,f
        result = tf.reduce_max(further_slices,2) #B,NEWSAMPLING_N,F
        return result

    
    def slice_adj(self,adj,l,c):
        w = tf.map_fn(lambda param:w[param],elems = l,dtype=tf.float32)
        w = tf.transpose(w)
        w = tf.map_fn(lambda param:w[param],elems = c,dtype=tf.float32)
        w = tf.transpose(w)
        return w