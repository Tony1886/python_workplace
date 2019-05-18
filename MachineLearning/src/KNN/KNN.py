#coding:utf-8
'''
Created on 2017��5��18��
������㷨
ʵ��ѧϰ(instance-based learning)
����ѧϰ(lazy learning)
@author: Administrator
'''

import os
import math
import random



# print(pointDistance(18, 90, 98, 2))
#导入数据
# 其中trainRation为训练比例
# testset为测试集
# trainset为训练集
def loadData(filename,trainRatio,testset=[],trainset=[]):
    f = open(filename,'rb')
    content = f.readlines()#一行一行读取数据
    # print(content[0])
    for i in range(1,len(content)):
        row = content[i-1].decode().split(',')#以逗号将数据分成列表类型，二进制解码
#         if i==1:
#             print(row)
        featureOnerow=[]
        for x in range(len(row)):
            featureOnerow.append(row[x])#读取一行数据 
        if random.random()<trainRatio:
            trainset.append(featureOnerow)#将一行数据加入到训练集
        else:
            testset.append(featureOnerow)#将一行数据加入测试机集
#     print(testset)

#寻找两个点的距离
def pointDistance(point1,point2):
    s=0
    for i in range(len(point1)):
        s+=math.pow((float(point1[i])-float(point2[i])),2)
    return math.sqrt(s)

#寻找最小的几个值
def findKmin(listData,k):
    minIndex=[]
    for i in range(k):
        minIndex.append(listData.index(min(listData)))
        listData[listData.index(min(listData))]=math.inf
    return minIndex

#找到最近邻的k个值
def getneighbor(onetestset,trainset,k):
    distance=[]
    for onetrainset in trainset:
        distance.append(pointDistance(onetestset[:len(onetestset)-1],onetrainset[:len(onetrainset)-1]))
    return findKmin(distance,k)

#近邻的几个值得classValue
def getpredict(onetestset,trainset,k):
    neighbors = getneighbor(onetestset,trainset,k)
    classValue={}#构造字典
    for index in neighbors:
        response = trainset[index][-1]
        if response in classValue:
            classValue[response]+=1
        else:
            classValue[response]=1
    dict=sorted(classValue.items(),key=lambda d:d[1],reverse=True)#reverse=True代表从大到小排列
    return dict[0][0]
            
filename=r'E:\BaiduYunDownload\麦子学院深度学习\代码与素材(1)\02KNN\irisdata.txt' 
trainRatio=0.8  
testset=[]
trainset=[] 
loadData(filename,trainRatio,testset,trainset)

print(repr(len(testset)))
print(len(testset))
k=3
neighbors=[]
correctNum=0
for onetestset in testset:
#     predict = getpredict(testset[i],trainset,k)
    predict = getpredict(onetestset,trainset,k)
#     print(predict)

    if predict == onetestset[-1]:
        correctNum+=1

print('正确率：%.2f'%(correctNum/len(testset)*100))    
    

      
        
        
    
    
    
    
    