#coding:utf-8
'''
Created on 2017��5��11��

@author: Administrator
'''
from numpy import *
# from matplotlib import *
import matplotlib.pyplot as plt
# from tkinter

x=linspace(-pi, pi, 100)
y=sin(x)
plt.plot(x,y)
plt.show()