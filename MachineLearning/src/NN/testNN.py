#coding:utf-8
'''
Created on 2017年5月30日
神经网络识别亦或运算
@author: Administrator
'''

from NN1 import NeuralNetwork

import numpy as np

nn = NeuralNetwork([2,3,1], 'tanh')     
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])     
y = np.array([0, 1, 1, 0])     
nn.fit(X, y)     
for i in [[0, 0], [0, 1], [1, 0], [1,1]]:    
    print(i, nn.predict(i))
