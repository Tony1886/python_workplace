#coding:utf-8
'''
Created on 2017��5��11��

@author: Administrator
'''
from math import *
from numpy import *
from matplotlib import *
import matplotlib.pyplot as plt
from numpy.dual import fft
from numpy.fft.helper import fftshift
from numpy.core.function_base import linspace


# x=linspace(-pi, pi, 100)
# f=sin(x)
# F=fftshift(fft(f))
# # plt.subplot(121)
# plt.plot(abs(F))
# plt.show()

# plt.subplot(122)
# plt.plot(f)
# plt.show()

#二维数组进行操作
# a = zeros((3,4))
# # i=0
# # j=0
# n=0
# for x in range(0,3):
#     for y in range(0,4):
#         a[x][y] = 4*x+y+1
#         n+=1
#         print("第%d次计算"%n)
#         print(a)
# # print(a)


# a='alkj flaf'
# pare=a.split()
# print(pare)
# x=5
# y=225.456189
# print("x=%g,y=%.3f"%(x,y))

# x=linspace(-pi,pi,100)
# y1=sin(x)
# y2=cos(x)
# plt.plot(x,y1,'b.',x,y2,'r-')
# plt.show()

# import sys,random
# def compute(n):
#     i=0;s=0
#     while i<=n:
#         s+=random.random()
#         i+=1
#     return s/n
# n=500000
# print('average of %d random number is %g'%(n,compute(n)))

x=linspace(-10,10,1000)
y=x
[X,Y] = meshgrid(x,y)
w = 2
Gauss = exp(-(X**2+Y**2)/pow(w,2))
# plt.subplot(121)
# plt.imshow(Gauss) 
# plt.subplot(122)
D=2
y1=exp(-x**2/D**2)
plt.plot(x,y1)
# y2=zeros((1,1000))
# y2[where(abs(x)<=D)]=1
# plt.plot(x,y2,'r')
plt.show()



