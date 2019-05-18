# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:12:41 2018
some function I uesd
形参*X 表示输入数量不定，是一个tuple
@author: Tan Zhijie
"""
from typing import Any, Union

from numpy.fft import fft,ifft,fft2,ifft2,fftshift,ifftshift
import numpy as np
from numpy import pi,exp
import math
import time 

# 快速傅里叶变换，直接0频调整
def mfft(f):
    
    f = np.array(f)
    F = np.zeros(np.shape(f))
    if np.size(f,0)==1 or np.size(f,1)==1:
        F = fftshift(fft(ifftshift(f)))
    else:
        F = fftshift(fft2(ifftshift(f)))
    return F

# 快速逆傅里叶变换，调整0频
def mifft(f):
    f = np.array(f)
    F = np.zeros(np.shape(f))
    if np.size(f,0)==1 or np.size(f,1)==1:
        F = fftshift(ifft(ifftshift(f)))
    else:
        F = fftshift(ifft2(ifftshift(f)))
    return F

# 划分网格
# 
def myGrid(N):
    if np.size(N) == 1:
        pass
        if np.mod(N,2) == 0:
            x = np.linspace(-N/2,N/2-1,N)
        else:
            x = np.linspace(-(N-1)/2,(N-1)/2,N)
        return x
    else:
        Ny = N[0]
        Nx = N[1]
        x = myGrid(Nx)
        y = myGrid(Ny)
        [Y,X] = np.meshgrid(y,x)
    return (Y,X)

# 构造光场传播的类
class Diff:
#    类初初始化，得到4个参数
    def __init__(self,lamb = 532e-9,Lx = 1e-3,Ly = 1e-3,d = 30e-2):
        self.lamb = lamb   # 波长
        self.Lx = Lx
        self.Ly = Ly      # 尺寸
        self.d = d      # 衍射距离
        
        
    #衍射积分的SFFT方法
    def SFFT(Ui,opt,mode = 1):
        lamb = opt.lamb
        Lx = opt.Lx
        Ly = opt.Ly
        d = opt.d
        Ui = np.array(Ui)
    #    Ny,Nx = myGrid(np.shape(Ui))
        Ny,Nx = np.shape(Ui)
        Lox = Nx*lamb*d/Lx
        Loy = Ny*lamb*d/Ly #衍射面大小
        lox = Lox/Nx
        loy = Loy/Ny #衍射面分辨率
        
        xi = myGrid(Nx)*Lx/Nx
        yi = myGrid(Ny)*Ly/Ny
        
        xo = myGrid(Nx)*lox
        yo = myGrid(Ny)*loy
        
        [Yi,Xi] = np.meshgrid(yi,xi)    
        [Yo,Xo] = np.meshgrid(yo,xo)
        if mode == 1:
            Fresnel = exp(1j*pi/lamb/d*(pow(Xi,2)+pow(Yi,2)))
            phase = exp(1j*2*pi/lamb*d)/(1j*lamb*d)*exp(1j*pi/lamb/d*(pow(Xo,2)+pow(Yo,2)))
            Uo = mfft(np.multiply(Ui,Fresnel))
            Uo = np.multiply(Uo,phase)
            return (Uo,yo,xo)
        else:
            back_Fresnel = exp(-1j*pi/lamb/d*(pow(Xi,2)+pow(Yi,2)))
            back_phase = exp(-1j*2*pi/lamb*d)/(1j*lamb*d)*exp(-1j*pi/lamb/d*(pow(Xo,2)+pow(Yo,2)))
            Uo = mifft(np.multiply(Ui,back_phase))
            Uo = np.multiply(Uo,back_Fresnel)
            return (Uo,yi,xi)
            
       
    #衍射积分的角谱衍射方法
    def angDiff(Ui,opt,mode = 1):
        lamb = opt.lamb
        Lx = opt.Lx
        Ly = opt.Ly
        d = opt.d
        Ui = np.array(Ui)
        
        Ny,Nx = np.shape(Ui)
        
        # 物面尺度
        xi = myGrid(Nx)*Lx/Nx
        yi = myGrid(Ny)*Ly/Ny
        
        # 物体面频率信息
        xf = myGrid(Nx)*1/Lx
        yf = myGrid(Ny)*1/Ly
        [Yf,Xf] =np. meshgrid(yf,xf)
        
        H = exp(1j*2*pi/lamb*d*np.sqrt(1-pow(lamb*Xf,2)-pow(lamb*Yf,2)));
        if mode == 1:
            Uf = mfft(Ui)
            Uo = mifft(np.multiply(Uf,H))
            return (Uo,yi,xi)
        else:
            Uo = mifft(np.multiply(mfft(Ui),np.conj(H)))
            return(Uo,yi,xi)
     
    #衍射积分的分数傅里叶变换方法       
    def FRTDiff(Ui,opt,mode = 1):
        lamb = opt.lamb
        Lx = opt.Lx
        Ly = opt.Ly
        d = opt.d
        Ui = np.array(Ui)
        
        Ny,Nx = np.shape(Ui)
        # 物面坐标
        yi = myGrid(Ny)*Ly/Ny
        xi = myGrid(Nx)*Lx/Nx
        [Yi,Xi] = np.meshgrid(yi,xi)
        # 物面频谱坐标
        xf = myGrid(Nx)*1/Lx
        yf = myGrid(Ny)*1/Ly
        [Yf,Xf] = np.meshgrid(yf,xf)
        
        f1 = Lx**2/lamb/Nx
        sita = math.atan(d/f1)
        beta = math.cos(sita)
        p = sita/2/pi
        
        phase = 1j*exp(1j*pi*(1-p)/2)*exp(-1j*pi/lamb/f1*(pow(Xi,2)+pow(Yi,2))*math.tan(sita/2))
        FRT1 = exp(-1j*pi/lamb/f1*(pow(Xi,2)+pow(Yi,2))*math.tan(sita/2))
        FRT2 = exp(-1j*pi*lamb*f1*math.sin(sita)*(pow(Xf,2)+pow(Yf,2)))
        
        Lo = Lx/beta
        xo = myGrid(Nx)*Lo/Nx
        yo = myGrid(Ny)*Lo/Ny
        if mode == 1:
            Uo = np.multiply(mifft(np.multiply(mfft(np.multiply(Ui,FRT1)),FRT2)),phase)
            return (Uo,yo,xo)
        else:
            Uo = np.multiply(mifft(np.multiply(mfft(np.multiply(Ui,np.conj(phase))),np.conj(FRT2))),np.conj(FRT1))
            return (Uo,yi,xi)

# 构造一维传递函数
def myH(xo,xi,lamb,L,mode = 1):
    boshik=2*pi/lamb
    idelta_x=xo[1]-xo[0]
    Xo,Xi= np.meshgrid(xi,xo)
    if mode == 1:
        H=exp(1j*boshik*L)/(1j*lamb*L)*exp(-1j*boshik/(2*L)*np.power((Xo-Xi),2))*idelta_x
    else:
        H=np.divide(exp(1j*boshik*L),(1j*lamb*np.sqrt(np.power((Xo-Xi),2)+L**2)))*exp(-1j*boshik/(2*L)*np.power((Xo-Xi),2))*idelta_x
    return H
    
 
# 我的神经网络    
class myNN:
    def sigmod(x,mode = 1 ):
        if mode == 1:
            return 1/(1+np.exp(-x))
        else:
            return myNN.sigmod(x)*(1-myNN.sigmod(x))
    
    def ReLu(x,mode = 1):
        x = np.array(x)
        if mode == 1:
            a = x.copy()
            a[a<0] = 0 
            return a
        else:
            a = np.ones(np.shape(x))
            a[x<0] = 0
            return a
        
    def tanh(x,mode = 1):
        if mode == 1:
            return np.tanh(x)
        else:
            return 1.0 - np.tanh(x)*np.tanh(x)
    
    # 设定激活函数
    def activation(a,Z,mode = 1):
        if a == 1:
            return myNN.sigmod(Z,mode)
        elif a == 2:
            return myNN.tanh(Z,mode)
        elif a == 3:
            return myNN.ReLu(Z,mode)
    
    def __init__(self, layers,active,learn_rate = 0.5,echo = 1000): 
        self.echo = echo
        self.learn_rate = learn_rate
        self.layers = layers
        self.active = active
        self.L = len(active) # 总共层数  
        if len(self.layers)-self.L != 1:
            print('层数设置错误')
        # w和b初始化
        self.w = [0]*self.L
        self.b = [0]*self.L
        
        for i in range(self.L):
            self.w[i] = np.random.randn(self.layers[i+1],self.layers[i])*np.sqrt(1/self.layers[i+1])
            self.b[i] = np.random.rand(self.layers[i+1],1)*np.sqrt(1/self.layers[i+1])   
    
    # 定义向后传递函数
    def singleFit(self,X):
        A=[0]*(self.L+1)
        A[0] = X     # 向后传播初始化 
        Z = [0]*self.L
        for i in range(self.L):
            Z[i] = np.dot(self.w[i],A[i])+self.b[i]
            A[i+1] = myNN.activation(self.active[i],Z[i])
        return (A,Z)
        
    # 定义向后传播
    def backPropa(self,y,A,Z,lossFunction):
        # 向前传播
        dw = [0]*self.L
        db = [0]*self.L
        dA = [0]*self.L
        dZ = [0]*self.L
        # 计算误差
        if lossFunction == 1:
            err_echo = -np.multiply(y,np.log(A[-1]))-np.multiply(1-y,np.log(1-A[-1]))
            dA[-1] = (-y/A[-1]+(1-y)/(1-A[-1])) # 向后传播初始化
        else:        
            err_echo = abs(y-A[-1])**2
            dA[-1] = -(y-A[-1])
                
        for i in range(len(self.layers)-2,-1,-1):
            dZ[i] = dA[i]*myNN.activation(self.active[i],Z[i],2)
            dw[i] = 1/len(y)*np.dot(dZ[i],A[i].T)
            db[i] = 1/len(y)*np.sum(dZ[i],axis = 1,keepdims = True)
            dA[i-1] = np.dot(self.w[i].T,dZ[i])
        
        return (dZ,dw,db,dA,err_echo)
        

    # 训练过程，1个echo向后传播，向前传播
    def fit(self,X,y,lossFunction = 1):
        start_time = time.clock()
        self.err = []
        for k in range(self.echo):
            A,Z = self.singleFit(X)
            dZ,dw,db,dA,err_echo = self.backPropa(y,A,Z,lossFunction)     
            self.err.append(np.sum(err_echo))
            for i in range(self.L):
                self.w[i] = self.w[i]-self.learn_rate*dw[i]
                self.b[i] = self.b[i]-self.learn_rate*db[i]
        end_time = time.clock()
        print('运行时间：',str(end_time-start_time))                   
    
    # 预测
    def predict(self,X):
        A,Z = self.singleFit(X)
        return A[-1]