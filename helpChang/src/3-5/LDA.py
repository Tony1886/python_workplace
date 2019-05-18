# -*- coding: utf-8 -*- h
from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def calulate_w():
    X1 = array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
                [0.481, 0.149], [0.437, 0.211]])
    X0 = array([[0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
                [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042], [0.719, 0.103]])
    mean1 = array([mean(X1[:, 0]), mean(X1[:, 1])])#正例密度均值，含糖率均值
    mean0 = array([mean(X0[:, 0]), mean(X0[:, 1])])#反例密度均值，含糖率均值
    m1 = shape(X1)[0]
    sw = zeros(shape=(2, 2))
    #类内散度矩阵计算
    for i in range(m1):
        xsmean = mat(X1[i, :] - mean1)
        sw += xsmean.transpose() * xsmean
    m0 = shape(X0)[0]
    for i in range(m0):
        xsmean = mat(X0[i, :] - mean0)
        sw += xsmean.transpose() * xsmean
    #w计算
    w = (mean0 - mean1) * (mat(sw).I)
    print(w)
    return w

#画出数据集数据以及直线w
def plot(w):
    dataMat = array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
                [0.481, 0.149], [0.437, 0.211],[0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
                [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042], [0.719, 0.103]])
    labelMat = mat([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]).transpose()
    m = shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i, 0])
            ycord1.append(dataMat[i, 1])
        else:
            xcord2.append(dataMat[i, 0])
            ycord2.append(dataMat[i, 1])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-0.2, 0.8, 0.1)
    y = array((-w[0, 0] * x) / w[0, 1])
    print(shape(x))
    print(shape(y))
    plt.sca(ax)
    plt.plot(x, y)  # gradAscent
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.title('LDA')
    plt.show()


w = calulate_w()
plot(w)