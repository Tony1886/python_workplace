# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:04:25 2018
测试一个tensorflow
minimize w^2-10w+25
@author: tanzj
"""
import tensorflow as tf
import numpy as np


coefficient = np.array([[1.],[-10.],[25.]])
w = tf.Variable(0,dtype = tf.float32)
x = tf.placeholder(tf.float32,[3,1])

#cost = tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
#cost = w**2 - 10*w + 25

cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.02).minimize(cost) # 0.02步长

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))

session.run(train,feed_dict = {x:coefficient})
print(session.run(w))

for i in range(1000):
    session.run(train,feed_dict = {x:coefficient})
print(session.run(w))
