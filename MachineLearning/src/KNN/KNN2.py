#coding:utf-8
'''
Created on 2017��5��18��

@author: Administrator
'''

from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()#直接从数据中导入iris数据
# print(iris)

knn.fit(iris.data, iris.target)
predictLabel = knn.predict([0.1,0.2,0.3,0.4])
print(predictLabel)
# print(type(iris.data))