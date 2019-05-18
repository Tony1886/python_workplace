# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:38:41 2018

@author: Tan Zhijie
"""

import numpy as np
import myFunction as mf
import tensorflow as tf
import matplotlib.pyplot as plt

# 生成几个随机二值图进行傅里叶变换
M = 1
N = 64

m = 2000
n = M*N
Es = np.zeros((m,M,N))
Ir = np.zeros(np.shape(Es))
for i in range(m):
    Es[i] = np.random.randint(0,2,[M,N])
    Er = mf.mfft(Es[i])
    Ir[i] = abs(Er)**2

Y = np.reshape(Es,[m,n])
X = np.reshape(Ir,[m,n])
X = X/np.max(X)


def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# 构造神经网络的一层
def add_layer(inputs,in_size,out_size,activation = None):
    W = tf.Variable(tf.random_normal([in_size,out_size])/in_size,name = 'W')
#    W = tf.Variable(tf.zeros([in_size,out_size]),name = 'W')
    b = tf.Variable(tf.zeros([1,out_size]),name = 'b')
    Z = tf.matmul(inputs,W)+b
    if activation ==None:
        output = Z
    else:
        output = activation(Z)
    return output

# define input
xs = tf.placeholder(tf.float32,[None,n])
ys = tf.placeholder(tf.float32,[None,n])
keep_drop = tf.placeholder(tf.float32)
# add layer
layer = [n,10*n,10*n,n]
for i in range(len(layer)-1):
    if i == 0:
        l = add_layer(xs,layer[i],layer[i+1],activation=tf.nn.relu)
    elif i==len(layer)-2:    
        prediction = add_layer(l,layer[i],layer[i+1], activation=tf.sigmoid)
    else:
        l = add_layer(l,layer[i],layer[i+1],activation=tf.nn.relu)

# loss function    交叉熵
#loss = tf.reduce_mean(tf.reduce_sum(-ys*tf.log(prediction),
#                                    reduction_indices = [1]))
# loss function   mse        
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                    reduction_indices = [1]))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train,feed_dict = {xs:X,ys:Y,keep_drop:0.5})
    if i%50 == 0:
        print(compute_accuracy(X, Y))

# 比较任意一幅图结果
test = 0
result = sess.run(prediction,feed_dict={xs:X[test].reshape([1,n])})
plt.figure
#plt.subplot(121)
#plt.imshow(result.reshape([8,8]))
plt.scatter(np.linspace(1,64,64),result)
#plt.subplot(122)
#plt.imshow(Es[test])
plt.scatter(np.linspace(1,64,64),Y[test],c = 'r')
plt.show()