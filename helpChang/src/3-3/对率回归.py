# -*- coding:utf-8 -*-
from sklearn import linear_model,datasets
import numpy as np
from  matplotlib import pyplot as plt
file1 = open('watermelon.txt','r')
data = [line.strip('\n').split(',') for line in file1]
X = [[float(raw[-3]), float(raw[-2])] for raw in data[1:]]
Y = [1 if raw[-1]=='是' else 0 for raw in data[1:]]
#读入数据,结果用1(好瓜)和0(坏瓜)表示
print(X)
print('------------------------------------------------------------------')
print(Y)
print('------------------------------------------------------------------')
X = np.array(X)
#将列表数据转换成矩阵数据方便numpy处理
print(X)
print('------------------------------------------------------------------')
h = .002
#网格步长

logreg = linear_model.LogisticRegression(C=1e5)

#使用sklearn中的LogisticRegression算法,传入参数C,越小表示更强的正则化
logreg.fit(X, Y)
#拟合

#绘制边界,所有点都在x的最大值和最小值,y的最大值和最小值所限定的范围内
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
test = np.c_[xx.ravel(), yy.ravel()]
print(test)
print('------------------------------------------------------------------')
Z = logreg.predict(test)
print(Z)
print('------------------------------------------------------------------')
#np.c_() 连接多个数组为一个数组
#logreg.predict()预测给定模型的x的目标值
#将结果放入一个彩色图中
Z = Z.reshape(xx.shape)
print(Z)
print('------------------------------------------------------------------')
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

#x轴为密度,y轴表示甜度
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Density')
plt.ylabel('Sugar rate')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
#绘制
plt.show()
