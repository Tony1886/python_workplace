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

#x = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])     
x = np.array([[0,0], [0,1], [1,0], [1,1]]) 
y = np.array([0, 1, 1, 0]) 

layer = [2,20,1]
active = [3,1]

# 输入变量变换
X = np.reshape(x.T,(len(x[0]),len(x)))
Y = np.reshape(y.T,(1,len(y)))

lossFunction = 1 # 1:-{ylog(a)+(1-y)log(1-a)} 2: 1/2*(a-y)**2 (该情况下非凸优，容易陷入局部最优解)
echo = 1000
learn_rate = 0.2

L = len(active) # 总共层数    

def sigmod(x,mode = 1 ):
    if mode == 1:
        return 1/(1+np.exp(-x))
    else:
        return sigmod(x)*(1-sigmod(x))

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
        return sigmod(Z,mode)
    elif a == 2:
        return tanh(Z,mode)
    elif a == 3:
        return ReLu(Z,mode)


# w和b初始化
w = [0]*L
b = [0]*L

for i in range(L):
    w[i] = np.random.randn(layer[i+1],layer[i])*np.sqrt(1/layer[i+1])
    b[i] = np.random.rand(layer[i+1],1)*np.sqrt(1/layer[i+1])   # 实际上会有些初值敏感的问题


start_time = time.clock()
err = []
for k in range(echo):

    # 向后传播
    A=[0]*(L+1)
    A[0] = X     # 向后传播初始化   
    Z = [0]*L
    for i in range(L):
        Z[i] = np.dot(w[i],A[i])+b[i]
        A[i+1] = activation(active[i],Z[i])
        
    # 向前传播
    dw = [0]*L
    db = [0]*L
    dA = [0]*L
    dZ = [0]*L
    # 计算误差
    if lossFunction == 1:
        err_echo = -np.multiply(y,np.log(A[-1]))-np.multiply(1-y,np.log(1-A[-1]))
        dA[-1] = (-y/A[-1]+(1-y)/(1-A[-1])) # 向后传播初始化
    else:        
        err_echo = abs(y-A[-1])**2
        dA[-1] = -(y-A[-1])
    err.append(np.sum(err_echo)) 
    
    for i in range(len(layer)-2,-1,-1):
        dZ[i] = dA[i]*activation(active[i],Z[i],2)
        dw[i] = 1/len(y)*np.dot(dZ[i],A[i].T)
        db[i] = 1/len(y)*np.sum(dZ[i],axis = 1,keepdims = True)
        dA[i-1] = np.dot(w[i].T,dZ[i])
    
    
    for i in range(L):
        w[i] = w[i]-learn_rate*dw[i]
        b[i] = b[i]-learn_rate*db[i]
        

end_time = time.clock()
print('运行时间：',str(end_time-start_time))                   
# 预测
Z = []
#A.append(np.array([0,1]).T)
for i in range(len(layer)-1):
    Z.append(np.dot(w[i],A[i])+b[i])
    A.append(activation(active[i],Z[i]))
print(A[-1])

plt.plot(err)
plt.show()