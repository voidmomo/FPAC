# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import os
import plotly.graph_objects as go
import h5py
import time

print('This process may take long time')


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

path = os.path.dirname('.')
train_path = os.path.join(path, "PrepData")
filenames = [d for d in os.listdir(train_path)]
print(train_path)
print(filenames)
train_points = None
train_labels = None
for d in filenames:
    cur_points, cur_labels = load_h5(os.path.join(train_path, d))
    cur_points = cur_points.reshape(1, -1, 3)
    cur_labels = cur_labels.reshape(1, -1)
    if train_labels is None or train_points is None:
        train_labels = cur_labels
        train_points = cur_points
    else:
        train_labels = np.hstack((train_labels, cur_labels))
        train_points = np.hstack((train_points, cur_points))
X_train = train_points.reshape(-1, num_points, 3)
train_labels_r = train_labels.reshape(-1, 1)

# load test points and labels
test_path = os.path.join(path, "PrepData_test")
filenames = [d for d in os.listdir(test_path)]
print(test_path)
print(filenames)
test_points = None
test_labels = None
for d in filenames:
    cur_points, cur_labels = load_h5(os.path.join(test_path, d))
    cur_points = cur_points.reshape(1, -1, 3)
    cur_labels = cur_labels.reshape(1, -1)
    if test_labels is None or test_points is None:
        test_labels = cur_labels
        test_points = cur_points
    else:
        test_labels = np.hstack((test_labels, cur_labels))
        test_points = np.hstack((test_points, cur_points))
X_test = test_points.reshape(-1, num_points, 3)
test_labels_r = test_labels.reshape(-1, 1)


# label to categorical
Y_train = tf.keras.utils.to_categorical(train_labels_r, k)
Y_test = tf.keras.utils.to_categorical(test_labels_r, k)

numpy.save('.\\ModelNet40\\X_train',X_train)
numpy.save('.\\ModelNet40\\X_test',X_test)
numpy.save('.\\ModelNet40\\Y_train',Y_train)
numpy.save('.\\ModelNet40\\Y_test',Y_test)
# %%

X_train = np.load('.\\ModelNet40\\X_train.npy')
X_test = np.load('.\\ModelNet40\\X_test.npy')
Y_train = np.load('.\\ModelNet40\\Y_train.npy')
Y_test = np.load('.\\ModelNet40\\Y_test.npy')


# %%
# 本段内容为给单个2048点云进行FPS采样
def farthest_point_sampling(pointcloud,npoint = 512):
    x = np.expand_dims(pointcloud, axis=0)    
    x = np.subtract(x, np.transpose(x, [1,0,2]))
    x = np.square(x)
    x = np.sum(x, axis=2) #按照坐标的一轴进行相加

    result = np.zeros((npoint,), dtype=np.int)

    farthest = np.random.randint(0,2048)

    for i in range(npoint):
        result[i] = farthest
        x[farthest,:] = -1
        dists = []
        farthest = np.argmax(x[:,result[0:i+1]].min(1))
    return result


# %%
time_start=time.time()
trainSampling512 = np.array(list(map(farthest_point_sampling,X_train)))
np.save('.\\ModelNet40Sampling\\train512idx',trainSampling512) 
time_end=time.time()
print('trainSampling512 totally cost',time_end-time_start)

time_start=time.time()
testSampling512 = np.array(list(map(farthest_point_sampling,X_test)))
np.save('.\\ModelNet40Sampling\\test512idx',testSampling512) 
time_end=time.time()
print('testSampling512 totally cost',time_end-time_start)


# %%
def get_sampling_points(idx,pointcloud):
    return pointcloud[idx]


# %%
trainSampling512xyz = []
for s in range(9840):
    c = get_sampling_points(trainSampling512[s],X_train[s])
    trainSampling512xyz.append(c)
trainSampling512xyz = np.array(trainSampling512xyz)
np.save('.\\ModelNet40Sampling\\train512point',trainSampling512xyz) 

testSampling512xyz = []
for s in range(2468):
    c = get_sampling_points(testSampling512[s],X_test[s])
    testSampling512xyz.append(c)
testSampling512xyz = np.array(testSampling512xyz)
np.save('.\\ModelNet40Sampling\\test512point',testSampling512xyz) 


# %%
# 抓取128采样相对于512点的idx信息
def farthest_point_sampling_128(pointcloud,npoint = 128):
    x = np.expand_dims(pointcloud, axis=0)    
    x = np.subtract(x, np.transpose(x, [1,0,2]))
    x = np.square(x)
    x = np.sum(x, axis=2) #按照坐标的一轴进行相加

    result = np.zeros((npoint,), dtype=np.int)

    farthest = np.random.randint(0,512)

    for i in range(npoint):
        result[i] = farthest
        x[farthest,:] = -1
        dists = []
        farthest = np.argmax(x[:,result[0:i+1]].min(1))
    return result

def get_sampling128_real_idx(idx,pointidx512):
    return pointidx512[idx]


# %%
time_start=time.time()
trainSampling128 = np.array(list(map(farthest_point_sampling_128,trainSampling512xyz)))
trainpoint128idx = []
for s in range(9840):
    c = get_sampling128_real_idx(trainSampling128[s],trainSampling512[s])
    trainpoint128idx.append(c)
trainpoint128idx = np.array(trainpoint128idx)
np.save('.\\ModelNet40Sampling\\train128idx',trainpoint128idx)
time_end=time.time()
print('trainSampling128 totally cost',time_end-time_start)


# %%
time_start=time.time()
testSampling128 = np.array(list(map(farthest_point_sampling_128,testSampling512xyz)))
testpoint128idx = []
for s in range(2468):
    c = get_sampling128_real_idx(testSampling128[s],testSampling512[s])
    testpoint128idx.append(c)
testpoint128idx = np.array(testpoint128idx)
np.save('.\\ModelNet40Sampling\\test128idx',testpoint128idx) 
time_end=time.time()
print('trainSampling128 totally cost',time_end-time_start)


# %%
trainSampling128xyz = []
for s in range(9840):
    c = get_sampling_points(trainpoint128idx[s],X_train[s])
    trainSampling128xyz.append(c)
trainSampling128xyz = np.array(trainSampling128xyz)
np.save('.\\ModelNet40Sampling\\train128point',trainSampling128xyz) 


# %%
testSampling128xyz = []
for s in range(2468):
    c = get_sampling_points(testpoint128idx[s],X_test[s])
    testSampling128xyz.append(c)
testSampling128xyz = np.array(testSampling128xyz)
np.save('.\\ModelNet40Sampling\\test128point',testSampling128xyz) 


# %%
# 抓取32采样相对于128点的idx信息
def farthest_point_sampling_32(pointcloud,npoint = 32):
    x = np.expand_dims(pointcloud, axis=0)    
    x = np.subtract(x, np.transpose(x, [1,0,2]))
    x = np.square(x)
    x = np.sum(x, axis=2) #按照坐标的一轴进行相加

    result = np.zeros((npoint,), dtype=np.int)

    farthest = np.random.randint(0,128)

    for i in range(npoint):
        result[i] = farthest
        x[farthest,:] = -1
        dists = []
        farthest = np.argmax(x[:,result[0:i+1]].min(1))
    return result

def get_sampling32_real_idx(idx,pointidx128):
    return pointidx128[idx]


# %%
time_start=time.time()
trainSampling32 = np.array(list(map(farthest_point_sampling_32,trainSampling128xyz)))
trainpoint32idx = []
for s in range(9840):
    c = get_sampling32_real_idx(trainSampling32[s],trainpoint128idx[s])
    trainpoint32idx.append(c)
trainpoint32idx = np.array(trainpoint32idx)
np.save('.\\ModelNet40Sampling\\train32idx',trainpoint32idx)

trainSampling32xyz = []
for s in range(9840):
    c = get_sampling_points(trainpoint32idx[s],X_train[s])
    trainSampling32xyz.append(c)
trainSampling32xyz = np.array(trainSampling32xyz)
np.save('.\\ModelNet40Sampling\\train32point',trainSampling32xyz) 
time_end=time.time()
print('trainSampling128 totally cost',time_end-time_start)


# %%
time_start=time.time()
testSampling32 = np.array(list(map(farthest_point_sampling_32,testSampling128xyz)))
testpoint32idx = []
for s in range(2468):
    c = get_sampling32_real_idx(testSampling32[s],testpoint128idx[s])
    testpoint32idx.append(c)
testpoint32idx = np.array(testpoint32idx)
np.save('.\\ModelNet40Sampling\\test32idx',testpoint32idx) 

testSampling32xyz = []
for s in range(2468):
    c = get_sampling_points(testpoint32idx[s],X_test[s])
    testSampling32xyz.append(c)
testSampling32xyz = np.array(testSampling32xyz)
np.save('.\\ModelNet40Sampling\\test32point',testSampling32xyz) 
time_end=time.time()
print('trainSampling128 totally cost',time_end-time_start)


# %%
# 抓取8采样相对于32点的idx信息
def farthest_point_sampling_8(pointcloud,npoint = 8):
    x = np.expand_dims(pointcloud, axis=0)    
    x = np.subtract(x, np.transpose(x, [1,0,2]))
    x = np.square(x)
    x = np.sum(x, axis=2) #按照坐标的一轴进行相加

    result = np.zeros((npoint,), dtype=np.int)

    farthest = np.random.randint(0,32)

    for i in range(npoint):
        result[i] = farthest
        x[farthest,:] = -1
        dists = []
        farthest = np.argmax(x[:,result[0:i+1]].min(1))
    return result

def get_sampling8_real_idx(idx,pointidx32):
    return pointidx32[idx]


# %%
time_start=time.time()
trainSampling8 = np.array(list(map(farthest_point_sampling_8,trainSampling32xyz)))
trainpoint8idx = []
for s in range(9840):
    c = get_sampling8_real_idx(trainSampling8[s],trainpoint32idx[s])
    trainpoint8idx.append(c)
trainpoint8idx = np.array(trainpoint8idx)
np.save('.\\ModelNet40Sampling\\train8idx',trainpoint8idx)

trainSampling8xyz = []
for s in range(9840):
    c = get_sampling_points(trainpoint8idx[s],X_train[s])
    trainSampling8xyz.append(c)
trainSampling8xyz = np.array(trainSampling8xyz)
np.save('.\\ModelNet40Sampling\\train8point',trainSampling8xyz) 
time_end=time.time()
print('trainSampling128 totally cost',time_end-time_start)


# %%
time_start=time.time()
testSampling8 = np.array(list(map(farthest_point_sampling_8,testSampling32xyz)))
testpoint8idx = []
for s in range(2468):
    c = get_sampling8_real_idx(testSampling8[s],testpoint32idx[s])
    testpoint8idx.append(c)
testpoint8idx = np.array(testpoint8idx)
np.save('.\\ModelNet40Sampling\\test8idx',testpoint8idx) 

testSampling8xyz = []
for s in range(2468):
    c = get_sampling_points(testpoint8idx[s],X_test[s])
    testSampling8xyz.append(c)
testSampling8xyz = np.array(testSampling8xyz)
np.save('.\\ModelNet40Sampling\\test8point',testSampling8xyz) 
time_end=time.time()
print('trainSampling128 totally cost',time_end-time_start)


# %%



