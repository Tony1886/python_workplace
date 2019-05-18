# coding=gbk 
# coding：utf-8
'''

 @Created on 2018年1月26日 下午2:43:08
 @author: Administrator
'''
from myFunc import mfft, mifft
from math import *
from numpy import *
# import numpy as np
from matplotlib import *
import matplotlib.pyplot as plt
from numpy.dual import fft
from numpy.fft.helper import fftshift
from numpy.core.function_base import linspace


x=linspace(-pi, pi, 100)
print(type(x))
print(shape(x))
y = random.random(size=(1,10))
print(type(y))
print(type(shape(y)))
print(type(shape(y)))
f=sin(x)
F=fftshift(fft(f))
#F = mfft(f)
plt.subplot(121)
plt.plot(abs(F))
plt.show()