#coding:utf-8
'''
Created on 2017��5��12��
决策树算法
@author: Administrator
'''
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsDate = open(r'E:\BaiduYunDownload\麦子学院深度学习\代码与素材(1)\01DTree\AllElectronics.csv',newline='')#在路径前加r
reader = csv.reader(allElectronicsDate)
headers =reader.__next__()#先读取第一行
# print(headers)
    
featureList=[]
labelList=[]
for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1,len(row)-1):
        rowDict[headers[i]] = row[i] #字典数据的存储
    featureList.append(rowDict)#将字典数据放入列表中
#print(featureList)

#vector the features
vec = DictVectorizer()#特征向量化
dummyX = vec.fit_transform(featureList) .toarray()
print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

# vectorize class labels
# print("labelList: " + str(labelList))
lb = preprocessing.LabelBinarizer()#类标签向量化
dummyY = lb.fit_transform(labelList)
# print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))


# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

#预测数据
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))
 
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))
 
predictedY = clf.predict(newRowX)
print("predictedY: " + str(predictedY))