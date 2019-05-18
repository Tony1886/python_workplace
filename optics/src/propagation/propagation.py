#coding:utf-8
'''
Created on 2017��5��21��
矩形光束的传播
@author: Administrator
'''

from numpy import *
import matplotlib.pyplot as plt

def Hfunction(xs,xr,lamb,dr):
#菲涅尔传递函数
    boshik = 2*pi/lamb
    [Xr,Xs]=meshgrid(xr,xs)    
    X=pow(Xs-Xr,2)
    H=exp(-1j*pi/(lamb*dr)*X)
    return H
    


dr=1
lamb=532e-9
Ls=5e-3
Lr=10e-3
Ns=100
Nr=100
xs=linspace(-Ls/2, Ls/2, Ns)
xr=linspace(-Lr/2, Lr/2, Nr)
# print(xs)
Es=zeros((1,Ns))
Es[0][int((Ns-1)/2-4):int((Ns-1)/2+4)]=1
H=Hfunction(xs, xr, lamb, dr)
Er=dot(Es,H)

plt.plot(xr,abs(Er[0]))
plt.show()