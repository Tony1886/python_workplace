# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:55:41 2018

@author: Tan Zhijie
"""

import tensorflow as tf

a = tf.constant([[2.,3.,-4.,5.],[3,4,5,2]])
sess = tf.Session()
print(sess.run(tf.clip_by_value(a,3,4)))
print(sess.run(tf.contrib.layers.l1_regularizer(.5)(a)))
print(sess.run(tf.contrib.layers.l2_regularizer(.5)(a)))