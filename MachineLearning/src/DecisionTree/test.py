#coding:utf-8
'''
Created on 2017��5��18��
字典数据的创建
@author: Administrator
'''


a=['one','two','three','four']
print(type(a))
print(len(a))
dirc={}
for i in range(1,len(a)):
    dirc[i]=a[i-1]
print(dirc)