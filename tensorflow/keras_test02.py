# -*- coding: utf-8 -*-
# @Time    : 2018/8/31 15:50
# @Author  : Tan Zhijie
# @Email   : tanzj@siom.ac.cn
# @File    : keras_test02.py
# @Software: PyCharm
# test FT 1D

from keras.models import Sequential
from keras.layers import Dense
from matplotlib.pylab import plot,show
import numpy as np

N = 100
K = 100000
signal = np.random.rand(K, N)
# measurement = abs(np.fft.fft(signal,axis = 0))**2
mat = np.random.rand(N, N)
measurement = np.dot(signal,mat)
# measurement = np.fft.fft(signal,axis = 0)

model = Sequential()
model.add(Dense(units = 200, input_shape = (N,), activation='relu'))
model.add(Dense(units = 4000, input_dim = 200, activation='relu'))
model.add(Dense(units = 200, input_dim = 4000, activation='relu'))
model.add(Dense(units = 100, input_dim = 200, activation='sigmoid'))

model.compile(optimizer='sgd',loss='mean_squared_error')

model.fit(measurement,signal,epochs=2)

x = np.random.rand(1, N)
y = np.dot(x,mat)
predict_x = model.predict_classes(y)
plot(np.transpose(x))
plot(np.transpose(predict_x),'r')
show()





