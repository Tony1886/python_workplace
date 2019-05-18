# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:55:08 2018

@author: tanzj

a test for machine learnning

"""

from sklearn import datasets,svm
import matplotlib.pyplot as plt  

digits = datasets.load_digits()
clf = svm.SVC(gamma = 0.001,C = 100.)
clf.fit(digits.data[:-1],digits.target[:-1])
predict = clf.predict(digits.data[-1])

fig = plt.figure()
plt.imshow(digits.data[-1].reshape(8,8))
 

