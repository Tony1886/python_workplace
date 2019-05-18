# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:36:36 2018
deep learning 
深层神经网络
@author: Tan Zhijie
"""


import numpy as np
import matplotlib.pyplot as plt 
import time 

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
    
'''    
#x = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])     
x = np.array([[0,0], [0,1], [1,0], [1,1]]) 
y = np.array([0, 1, 1, 0]) 
# 输入变量变换
X = np.reshape(x.T,(len(x[0]),len(x)))
Y = np.reshape(y.T,(1,len(y)))

layer = [2,3,1]
active = [3,1]


lossFunction = 1 # 1:-{ylog(a)+(1-y)log(1-a)} 2: 1/2*(a-y)**2 (该情况下非凸优，容易陷入局部最优解)
echo = 1000
learn_rate = 0.2
test = myNN(layer,active,learn_rate,echo)
test.fit(X,Y)
ans = test.predict(X)
print(ans)
'''

x = np.mat( '2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,1,2,2,2,1,1;\
            2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3;\
            3,3,3,3,3,3,2,3,2,3,1,1,2,2,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,3,1,1,2,3,2;\
            1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1;\
            0.697,0.774,0.634,0.668,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719;\
            0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103\
            ')
x = np.array(x)
y = np.mat('1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0')
y = np.array(y)

nx,mx = np.shape(x) # 得到输入变量数目和测试集数目
ny,my = np.shape(y)
layers = [8,10,1]
active = [2,1]

lossFunction = 1 
echo = 1000
learn_rate = 0.1
test = myNN(layers,active,learn_rate,echo)

test.fit(x,y)
ans = test.predict(x)

print(ans)
plt.plot(test.err)
plt.show
print('最后误差'+str(test.err[-1]))
