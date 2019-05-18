# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:16:13 2018
关于numpy的教程
@author: Tony
"""
import numpy as np
import time


# =============================================================================
# # np的运算比较慢
# 
# t0 = time.clock()
# v1 = 3.14
# for i in range(10000):
#     v1*v1
# t1 = time.clock()
# 
# v2 = np.float64(3.14)
# for i in range(10000):
#     v2*v2
# t2 = time.clock()
# print('第一个时间：',str(t1-t0))
# print('第一个时间：',str(t2-t1))
# =============================================================================


print(np.empty((2,3),np.int)) # 初始化

def func2(i,j):
    return (i+1)*(j+1)
print(np.fromfunction(func2,(9,9)))


np.convolve