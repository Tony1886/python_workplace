# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 15:21:52 2018
手写字符集2
@author: Tan Zhijie
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
#    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# 构造神经网络的一层
def add_layer(inputs,in_size,out_size,activation = None):
    W = tf.Variable(tf.random_normal([in_size,out_size])/in_size,name = 'W')
#    W = tf.Variable(tf.zeros([in_size,out_size]),name = 'W')
    b = tf.Variable(tf.zeros([1,out_size])+0.1,name = 'b')
    Z = tf.matmul(inputs,W)+b
    if activation ==None:
        output = Z
    else:
        output = activation(Z)
    return output

# define input
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

# add layer
#l1 = add_layer(xs,784,784,activation=tf.nn.relu)
prediction = add_layer(xs, 784, 10, activation=tf.nn.softmax)

loss = tf.reduce_mean(tf.reduce_sum(-ys*tf.log(prediction),
                                    reduction_indices = [1]))

train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train,feed_dict = {xs:batch_xs,ys:batch_ys})
    if i%100 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))