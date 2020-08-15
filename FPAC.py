import tensorflow.keras.backend as K
import tensorflow as tf 
from tensorflow.keras.layers import Layer, BatchNormalization
from tensorflow.keras import regularizers
import numpy as np
import Mish


class FPAC_Layer(Layer):
    def __init__(self, xyz, cin, cout, m1, m2, mr, mid, maxn ,framepoints, numframe, N, knn=True, l2=None, dtype=tf.float32):
        super(FPAC_Layer, self).__init__()
        self.xyz = xyz
        self.cin = cin
        self.cout = cout
        self.m1 = m1
        self.m2 = m2
        self.mr = mr
        self.mid = mid
        self.maxn = maxn
        self.framepoints = framepoints
        self.numframe = numframe
        self.N = N
        self.l2 = l2
        self.bn = BatchNormalization()
        self.m1_filters = []
        self.m1_b = []
        self.m2_filters = []
        self.m2_b = []
        self.mr_filters = []
        self.mr_b = []
        self.m1_bn = []
        self.m1_num = len(self.m1) - 1
        self.mish = Mish()
        self.m2_num = len(self.m2) - 1
        self.mr_num = len(self.mr) - 1
        self.knn = knn
        for i in range(self.m1_num):
            self.m1_bn.append(BatchNormalization())
        self.m2_bn = []
        for i in range(self.m2_num):
            self.m2_bn.append(BatchNormalization())
        self.mr_bn = []
        for i in range(self.mr_num):
            self.mr_bn.append(BatchNormalization())
        self.m1[0] = 3
        self.m1[-1] = 1
        self.m2[0] = self.cin * self.cout
        self.m2[-1] = self.mid
        self.mr[0] = self.cin * self.mid
        self.mr[-1] = self.cout



    def build(self,input_shape):
        self.framepointsweights = self.add_weight(self.name + '_framepointsWeights',shape = [self.numframe,self.cin,self.cout],initializer='glorot_uniform',dtype = self.dtype)
        # m1
        for i in range(self.m1_num):
            temp_filter = self.add_weight(self.name + '_m1_filters_' + str(i),shape = [self.m1[i],self.m1[i+1]],initializer = 'glorot_uniform', regularizer=regularizers.get(self.l2), dtype = self.dtype)
            temp_b = self.add_weight(self.name + '_m1_b_' + str(i),shape = [self.m1[i+1]],initializer = 'glorot_uniform', dtype = self.dtype)
            self.m1_filters.append(temp_filter)
            self.m1_b.append(temp_b)

        # m2
        for i in range(self.m2_num):
            temp_filter = self.add_weight(self.name + '_m2_filters_' + str(i),shape = [self.m2[i],self.m2[i+1]],initializer = 'glorot_uniform', regularizer=regularizers.get(self.l2), dtype = self.dtype)
            temp_b = self.add_weight(self.name + '_m2_b_' + str(i),shape = [self.m2[i+1]],initializer = 'glorot_uniform', dtype = self.dtype)
            self.m2_filters.append(temp_filter)
            self.m2_b.append(temp_b)

        # mr
        for i in range(self.mr_num):
            temp_filter = self.add_weight(self.name + '_mr_filters_' + str(i),shape = [self.mr[i],self.mr[i+1]],initializer = 'glorot_uniform', regularizer=regularizers.get(self.l2), dtype = self.dtype)
            temp_b = self.add_weight(self.name + '_mr_b_' + str(i),shape = [self.mr[i+1]],initializer = 'glorot_uniform', dtype = self.dtype)
            self.mr_filters.append(temp_filter)
            self.mr_b.append(temp_b)
        
        super(FPAC_Layer, self).build(input_shape)
    def call(self, input):
        # Get the IDX of the nearest maxn points
        x = tf.expand_dims(input, axis=1)
        x = tf.subtract(x, tf.transpose(x, [0, 2, 1, 3]))
        x = tf.square(x)
        x = tf.reduce_sum(x, axis=3) 
        
        _,idx = tf.nn.top_k(-x,self.maxn) #_:B,N,n   idx:B,N,n
        idx = tf.expand_dims(idx,3) #idx:B,N,n,1
        xyz_slices = tf.map_fn(lambda param:tf.gather_nd(param[0],param[1]),elems=(self.xyz,idx),dtype=self.dtype)  #B,N,n,3
        further_slices = tf.map_fn(lambda param:tf.gather_nd(param[0],param[1]),elems=(input,idx),dtype=self.dtype)  #B,N,n,in
        xyz_slices = tf.reshape(xyz_slices,[-1,self.maxn,3])  # B*N,n,3

        center_points = xyz_slices[:,0,:] # B*N,3
        center_points = tf.expand_dims(center_points,1) # B*N,1,in

        slices = tf.subtract(xyz_slices,center_points) # B*N,n,3 - B*N,1,3  = B*N,n,3
        slices = tf.reshape(slices,[-1,3]) #B*N*n,3
        # Random rotation framepoints
        self.framepoints = self.rotate_frame_points(self.framepoints)

        slices = tf.concat([slices,self.framepoints],0)  #Here, in order to add the framepoint to participate in the calculation, the self.num_ Fixed point points can be used to calculate the increase in the loss function
        # B*N*n+v,3

        slices = tf.expand_dims(slices,1) # B*N*n+v,1,3    

        
        diff = tf.subtract(slices,self.framepoints)  #B*N*maxn+nf,num_fixedPoint,3   B*N*maxn+nf,1,3 - num_fixedPoint,3 = B*N*maxn+nf,num_fixedPoint,3

        x = tf.reshape(diff,[-1,3]) #(B*N*maxn+nf)*num_fixedPoint,3

        # m1
        for i in range(self.m1_num):
            x = self.mish(tf.matmul(x,self.m1_filters[i])+self.m1_b[i])
            x = self.m1_bn[i](x)



        w = tf.reshape(self.framepointsweights, [-1, self.cin*self.cout]) #num_fixedPoint,in_channels*out_channels

        # m2
        for i in range(self.m2_num):
            w = self.mish(tf.matmul(w,self.m2_filters[i])+self.m2_b[i])
            w = self.m2_bn[i](w)
        # m2
        l1 = w

        x = tf.reshape(x,[-1,self.numframe,1])  #(B*N*maxn+nf),num_fixedPoint,1
        w = x*w #[B*N*maxn+nf,固定点个数,32]

        w = tf.reshape(w,[-1,self.numframe,self.mid])
        w = tf.reduce_sum(w, 1) # [B*N*maxn+nf,mid]


        w = tf.reshape(w, [-1,self.mid])  # (B*N*maxn+nf),mid

        f = tf.reshape(further_slices,[-1,self.maxn,self.cin])  #B×N,MAXN,in
        f = tf.transpose(f,[0,2,1]) #B×N,in,MAXN
        l = w[-self.numframe:]   

        w = w[:-self.numframe]   #B×N×MAXN,mid
        w = tf.reshape(w,[-1,self.maxn,self.mid])  #BxN,MAXN,mid
        f = tf.matmul(f,w) #BxN,in,32  

        
        f = tf.reshape(f,[-1,self.cin*self.mid])  #B*N,in*mid
        # mr
        for i in range(self.mr_num):
            f = self.mish(tf.matmul(f,self.mr_filters[i])+self.mr_b[i])
            f = self.mr_bn[i](f)

        f = tf.reshape(f,[-1,self.N,self.cout])
        # mr
        l = tf.square(tf.subtract(l1,l)) #num_fixedPoint,mid
        l = tf.sqrt(tf.reduce_sum(tf.reduce_sum(l)))/self.numframe/32
        tf.compat.v1.add_to_collection('losse2',l)
        return f

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [(None, input_shape[1], out_channels)]

    # Rotate Frame Points
    def rotate_frame_points(self, framepoint):
        # define rotation angle
        rotation_angle = tf.random.uniform((1,)) * 2 * np.pi
        cosval = tf.cos(rotation_angle)
        sinval = tf.sin(rotation_angle)
        rotation_matrix = tf.Variable([[cosval, [0], sinval], [[0], [1], [0]], [-sinval, [0], cosval]])
        # the rotation matrix
        rotation_matrix = tf.squeeze(rotation_matrix)
        # rotate the frame point
        result = tf.matmul(framepoint,rotation_matrix)
        return result
