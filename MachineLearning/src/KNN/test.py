#coding:utf-8
'''
Created on 2017��5��18��

@author: Administrator
'''
# import math
# mylist = [[1,5,6],[5,7,8],[3,9,10],[4,11,12]]
# print(len(mylist[0]))
# a=[]
# for i in mylist:
#     a.append(i[0])
# print(a)
# b=sorted(a)
# print(b)
# print(a)
# 
# print(max(a))
# print(math.inf)

dic={'a':3,'b':4,'c':2,'d':2}
dict = sorted(dic.items(),key=lambda d:d[1])
print(dict)