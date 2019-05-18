# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:37:48 2018
使用tensorflow 创建 NN
演示tensorboard
@author: Tan Zhijie
"""

import tensorflow  as tf
import numpy as np
#import matplotlib.pyplot as plt

# 构造神经网络的一层
def add_layer(inputs,in_size,out_size,n_layer,activation_function = None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.random_normal([in_size,out_size]),name = 'W')
            tf.summary.histogram(layer_name+'/weight',W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([1,out_size])+0.1,name = 'b')
            tf.summary.histogram(layer_name+'/biases',b)
        with tf.name_scope('Z'):
            Z = tf.matmul(inputs,W)+b
        if activation_function is None:
            output = Z
        else:
            output = activation_function(Z)
        tf.summary.histogram(layer_name+'/outputs',output)    
    return output


x_data = np.linspace (-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')

l1 = add_layer(xs,1,10,n_layer = 1,activation_function = tf.nn.relu)
predition = add_layer(l1,10,1,n_layer = 2 ,activation_function = None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices = [1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(x_data,y_data)
#plt.ion()# show之后放置暂停
#plt.show()


sess = tf.Session()
merged= tf.summary.merge_all()
writer = tf.summary.FileWriter('logs')
writer.add_graph(sess.graph)

sess.run(init)
for i in range(1000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0 :
        rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(rs,i)
#        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
#        try:
#            ax.lines.remove(lines[0])
#        except Exception:
#            pass
#        prediction_value = sess.run(predition,feed_dict={xs:x_data,ys:y_data})
#        lines = ax.plot(x_data,prediction_value,'r-',lw = 5)
#        plt.pause(0.1)

