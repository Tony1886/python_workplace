# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:12:09 2018

@author: Tan Zhijie
"""

import tensorflow  as tf

'''
matrix1 = tf.constant([[3,2]])
matrix2 = tf.constant([[2],[3]])

product = tf.matmul(matrix1,matrix2) 
multi = tf.multiply(matrix1,tf.transpose(matrix2))
sess = tf.Session()
result1 = sess.run(product)
result2 = sess.run(multi)
print (result1)
print (result2)
sess.close()
'''

state = tf.Variable(0,name = 'counter')
print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)
updata = tf.assign(state,new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        print(sess.run(updata))
        

def add_layer(inputs,in_size,out_size,activation_function = None):
    W = tf.Variable(tf.random_normal([in_size,out_size]))
    b = tf.Variable(tf.zeros([1,out_size])+0.1)
    Z = tf.matmul(inputs,W)+b
    if activation_function ==None:
        output = Z
    else:
        output = activation_function(Z)
    return output