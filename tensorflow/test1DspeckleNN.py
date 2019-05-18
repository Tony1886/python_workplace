# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:45:12 2018

@author: Tony
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(r'F:\python\python_workplace\myFunc')
import myFunction as mf
import tensorflow as tf


lamb = 532e-9
d1 = 10e-2
d2 = 30e-2
Ls = 1e-3
Ns = 100

xs = mf.myGrid(Ns)*Ls/Ns

Lo = 1e-3
No = 10
xo = mf.myGrid(No)*Lo/No

lr = 5.86e-6
Nr = 256
xr = mf.myGrid(Nr)*lr

H1 = mf.myH(xo,xs,lamb,d1)
H2 = mf.myH(xr,xo,lamb,d2)

np.random.seed(1)
fai = np.random.rand(Ns,1)
Es = np.exp(1j*(fai-0.5)*2*np.pi)

Eo = np.matmul(H1,Es)
plt.figure(1)
plt.plot(np.power(np.abs(Eo),2))
plt.show()

# =============================================================================
# plt.figure(2)
# plt.imshow(np.power(np.abs(H1),2))
# plt.show()
# =============================================================================

K = 10000 # 训练次数
Eo = np.random.rand(No,K)
Er = np.matmul(H2,Eo)
Ir = np.abs(Er)**2
Ir = Ir.T
Ir = Ir/np.max(Ir)*4096
Eo = Eo.T
Eo = Eo/np.max(Eo)

xs = tf.placeholder(tf.float32,[None,Nr])
ys = tf.placeholder(tf.float32,[None,No])

# add layers
layers = [Nr,2*Nr,2*Nr,No]

l1_w = tf.Variable(tf.random_normal([layers[0],layers[1]])/layers[0])
l1_b = tf.Variable(tf.zeros([1,layers[1]]))

l1 = tf.nn.relu(tf.matmul(xs,l1_w) + l1_b)

l2_w = tf.Variable(tf.random_normal([layers[1],layers[2]])/layers[1])
l2_b = tf.Variable(tf.zeros([1,layers[2]]))
l2 = tf.nn.relu(tf.matmul(l1,l2_w) + l2_b)

l3_w = tf.Variable(tf.random_normal([layers[-2],layers[-1]])/layers[-2])
l3_b = tf.Variable(tf.zeros([1,layers[-1]]))
y = tf.nn.relu(tf.matmul(l2,l3_w) + l3_b)

loss = tf.losses.mean_squared_error(y,ys)

#train = tf.train.GradientDescentOptimizer(0.6).minimize(loss)
train = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 定义误差函数

for i in range(500):
    sess.run(train,feed_dict={xs:Ir,ys:Eo})
    if i%50 ==  0:
        print(sess.run(loss,feed_dict={xs:Ir,ys:Eo}))
        
        
## 使用一个随机信号测试学习结果\
test_Num = 1
Eo_test = np.random.rand(No,test_Num)
Er_test = np.matmul(H2,Eo_test)
Ir_test = np.abs(Er_test)**2

Ir_test = Ir_test.T/np.max(Ir_test)*4096
Eo_test = Eo_test.T/np.max(Eo_test)

prediction = sess.run(y,feed_dict={xs:Ir_test,ys:Eo_test})
print(sess.run(loss,feed_dict={xs:Ir_test,ys:Eo_test}))

show_Num = 1
plt.figure(2)
plt.plot(np.reshape(Eo_test[None,show_Num-1],[-1]))
plt.plot(np.reshape(prediction[None,show_Num-1],[-1]),'r')
plt.show()


