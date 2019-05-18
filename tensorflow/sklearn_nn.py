# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:38:55 2018
使用sklearn数据学习
@author: Tan Zhijie
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

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
xs = tf.placeholder(tf.float32,[None,64])# 8X8
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32) # 设置dropout 

# add layer
l1 = add_layer(xs,64,400,activation=tf.nn.tanh)
prediction = add_layer(l1, 400, 10, activation=tf.nn.softmax)

# loss function
loss = tf.reduce_mean(tf.reduce_sum(-ys*tf.log(prediction),
                                    reduction_indices = [1]))
# train
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train, feed_dict={xs: X_train, ys: y_train,keep_prob:0.5})
    if i%50 == 0:
        print(compute_accuracy(X_train, y_train),compute_accuracy(X_test, y_test))
        
        
        