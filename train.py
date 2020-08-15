#%%
import tensorflow as tf
import numpy as np
import os
import h5py
import time
from FPAC import FPAC_Layer
from PointPooling import PointPooling
from GatherThePoint import GatherThePoint
from Mish import Mish
from tensorflow.keras.layers import Input, Dense, MaxPooling1D, AveragePooling1D, Flatten, Lambda, Dropout, BatchNormalization, LeakyReLU, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import optimizers

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

#%%
X_train = np.load('.\\ModelNet40\\X_train.npy')
X_test = np.load('.\\ModelNet40\\X_test.npy')
Y_train = np.load('.\\ModelNet40\\Y_train.npy')
Y_test = np.load('.\\ModelNet40\\Y_test.npy')

X_train512xyz = np.load('.\\ModelNet40Sampling\\train512point.npy')
X_train128xyz = np.load('.\\ModelNet40Sampling\\train128point.npy')
X_train32xyz = np.load('.\\ModelNet40Sampling\\train32point.npy')
X_train8xyz = np.load('.\\ModelNet40Sampling\\train8point.npy')

X_train512idx = np.load('.\\ModelNet40Sampling\\train512idx.npy')
X_train128idx = np.load('.\\ModelNet40Sampling\\train128idx.npy')
X_train32idx = np.load('.\\ModelNet40Sampling\\train32idx.npy')
X_train8idx = np.load('.\\ModelNet40Sampling\\train8idx.npy')

X_test512idx = np.load('.\\ModelNet40Sampling\\test512idx.npy')
X_test128idx = np.load('.\\ModelNet40Sampling\\test128idx.npy')
X_test32idx = np.load('.\\ModelNet40Sampling\\test32idx.npy')
X_test8idx = np.load('.\\ModelNet40Sampling\\test8idx.npy')

X_test512xyz = np.load('.\\ModelNet40Sampling\\test512point.npy')
X_test128xyz = np.load('.\\ModelNet40Sampling\\test128point.npy')
X_test32xyz = np.load('.\\ModelNet40Sampling\\test32point.npy')
X_test8xyz = np.load('.\\ModelNet40Sampling\\test8point.npy')

#%%
def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

dtype=tf.float32
l2 = regularizers.l2(0.002)

B = 8
N = 2048
num_framepoint = 9
conff1 = 0.075  
conff2 = 0.130  
conff3 = 0.250  
conff4 = 0.4  
conff5 = 0.8  

framepoints = tf.convert_to_tensor([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[0,0,0]],dtype=dtype)

framepoints1 = framepoints*conff1
framepoints2 = framepoints*conff2
framepoints3 = framepoints*conff3
framepoints4 = framepoints*conff4
framepoints5 = framepoints*conff5

def categorical_crossentropy_with_losse2(y_true, y_pred, w=10):
    ls = tf.compat.v1.get_collection('losse2')
    ls = tf.add_n(ls)
    return categorical_crossentropy(y_true, y_pred)+w*ls

inputs = Input(shape=(2048,3))

idx512 = Input(shape=(512),dtype = tf.int32)
idx128 = Input(shape=(128),dtype = tf.int32)
idx32 = Input(shape=(32),dtype = tf.int32)
idx8 = Input(shape=(8),dtype = tf.int32)

xyz512 = GatherThePoint(inputs)(idx512)
xyz128 = GatherThePoint(inputs)(idx128)
xyz32 = GatherThePoint(inputs)(idx32)
xyz8 = GatherThePoint(inputs)(idx8)

c1 = Conv1D(64,1)(inputs)
x = FPAC_Layer(xyz = inputs, cin = 3, cout = 64, m1=[3,9,1], m2=[192,64,32], mr=[96,64], mid=32, maxn=32, framepoints=framepoints1,numframe=num_framepoint,N=2048, l2=l2, dtype=dtype)(inputs)
x = tf.add(x,c1)
x = PointPooling(batch_sample_xyz=inputs,sampling=xyz512, poolN=9)(x)
x = BatchNormalization()(x)

c2 = Conv1D(128,1)(x)
x = FPAC_Layer(xyz = xyz512, cin = 64, cout = 128, m1=[3,9,1], m2=[8192,64,32], mr=[2048,64], mid=32, maxn=32, framepoints=framepoints2,numframe=num_framepoint,N=2048, l2=l2, dtype=dtype)(x)
x = tf.add(x,c2)
x = PointPooling(batch_sample_xyz=xyz512,sampling=xyz128, poolN=9)(x)
x = BatchNormalization()(x)

c3 = Conv1D(512,1)(x)
x = FPAC_Layer(xyz = xyz128, cin = 128, cout = 512, m1=[3,9,1], m2=[65536,64,32], mr=[4096,64], mid=32, maxn=32, framepoints=framepoints3,numframe=num_framepoint,N=2048, l2=l2, dtype=dtype)(x)
x = tf.add(x,c3)
x = PointPooling(batch_sample_xyz=xyz128,sampling=xyz32, poolN=9)(x)
x = BatchNormalization()(x)

c4 = Conv1D(1024,1)(x)
x = FPAC_Layer(xyz = xyz32, cin = 512, cout = 1024, m1=[3,9,1], m2=[524288,64,32], mr=[16384,64], mid=32, maxn=32, framepoints=framepoints4,numframe=num_framepoint,N=2048, l2=l2, dtype=dtype)(x)
x = tf.add(x,c4)
x = PointPooling(batch_sample_xyz=xyz32,sampling=xyz8,poolN=8)(x)
x = BatchNormalization()(x)



x = MaxPooling1D(pool_size=8)(x)

x = Dense(512)(x)
x = Mish()(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.3)(x)
x = Dense(256)(x)
x = Mish()(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.3)(x)
x = Dense(128)(x)
x = Mish()(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.3)(x)
x = Dense(40, activation = 'softmax')(x)
prediction = Flatten()(x)


model = Model(inputs=[inputs,idx512,idx128,idx32,idx8], outputs=prediction)

print(model.summary())

#(model, to_file='model2.png', show_shapes=True)
model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
              loss=categorical_crossentropy_with_losse2,
              metrics=['accuracy'],
              experimental_run_tf_function=False)


#%%

mostright=0

for i in range(1,250):
    #model.fit(train_points_r, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
    # rotate and jitter the points
    train_points_rotate = rotate_point_cloud(X_train)
    train_points_jitter = jitter_point_cloud(train_points_rotate)
    s = "Current Epoch is:" + str(i)
    print(s)
    his = model.fit([train_points_jitter,X_train512idx,X_train128idx,X_train32idx,X_train8idx], Y_train, batch_size=16, epochs=1, shuffle=True, verbose=1)
    if i % 5 == 0:
        print('Evaluate Test')
        score = model.evaluate([X_test,X_test512idx,X_test128idx,X_test32idx,X_test8idx], Y_test, verbose=1)
        print('Test Loss: ', score[0])
        print('Test Accuracy: ', score[1])
        if(score[1]>mostright):
            mostright = score[1]
            modelname = '..\\saveweight\\fpac'+'_'+str(i)+'_'+str(score[1])+'.hdf5'
            model.save_weights(modelname)
        if(i % 50 == 0):
            modelname = '..\\saveweight\\fpac'+'_'+str(i)+'_'+str(score[1])+'.hdf5'
            model.save_weights(modelname)

# score the model
score = model.evaluate([X_test,X_test512idx,X_test128idx,X_test32idx,X_test8idx], Y_test, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save_weights('..\\saveweight\\final500.hdf5')



# %%
