# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:37:48 2018
使用tensorflow 创建 NN
@author: Tan Zhijie
"""

import tensorflow  as tf
import numpy as np
import matplotlib.pyplot as plt

# 构造神经网络的一层
def add_layer(inputs,in_size,out_size,activation_function = None):
    W = tf.Variable(tf.random_normal([in_size,out_size]))
    b = tf.Variable(tf.zeros([1,out_size])+0.1)
    Z = tf.matmul(inputs,W)+b
    if activation_function ==None:
        output = Z
    else:
        output = activation_function(Z)
    return output


x_data = np.linspace (-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
predition = add_layer(l1,10,1,activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices = [1]))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data,y_data)
plt.ion()# show之后放置暂停
plt.show()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train,feed_dict={xs:x_data,ys:y_data})
        if i%50 == 0 :
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(predition,feed_dict={xs:x_data,ys:y_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw = 5)
            plt.pause(0.1)
