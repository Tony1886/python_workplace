# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:26:25 2018
cvx test
@author: Tony
"""

import cvxpy as cvx
import numpy
#Problem data.

m = 10
n = 5
numpy.random.seed(1)
A = numpy.random.randn(m, n) #矩阵A
b = numpy.random.randn(m)    #向量b
#变量x，x是一个向量
x = cvx.Variable(n)
z = A*x

objective = cvx.Minimize(cvx.sum_squares(z - b))
constraints = [0 <= x, x <= 1]
prob = cvx.Problem(objective, constraints)
print("Optimal value", prob.solve())
print("Optimal var")
print(x.value)             # A numpy ndarray.
print(z.value)             # A numpy ndarray.
