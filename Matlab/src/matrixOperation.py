#coding:utf-8
'''
Created on 2017年5月21日
矩阵运算
@author: Administrator
'''
from numpy import *


#矩阵构造
a1=array([1,2,3])
print(a1)
a1=mat(a1)
print(a1)
a2=mat(zeros((2,3)))
print(a2)
a3=random.rand(2,2)
print(a3)
a4=random.randint(2,10,size=(4,3))#产生2-10的整数矩阵，10不包含
print(a4)
a5=[1,2,3]
a5=diag(a5)
print(a5)#构造对角矩阵
a6=eye(2, 3, dtype=float)
print(a6)

#矩阵运算
a1=mat([1,2]); #1X2矩阵     
a2=mat([[1],[2]]);#2X1矩阵
a3=a1*a2;
print(a1,'\n叉乘',a2,'\n等于',a3,'\n')

a1=mat([1,1]);
a2=mat([2,2]);
a3=multiply(a1,a2);
print(a1,'\n点乘',a2,'\n等于',a3,'\n')



x=linspace(-10, 10, 5,dtype=float)
y=linspace(-10, 10, 5,dtype=float)
[X,Y]=meshgrid(x,y)
print(Y)
print(X)