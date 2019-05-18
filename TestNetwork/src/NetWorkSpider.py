#coding:utf-8
'''
Created on 2017年5月11日
从网页上下载图片
@author: Administrator
'''
import re
import urllib.request
import datetime
import os
#得到一个网页代码
def getHtmlCode(htmlurl):
    _f = urllib.request.urlopen(htmlurl)
    htmlCode = _f.read().decode("utf-8")
    return htmlCode
#从网页代码中获得图片
def getImag(htmlCode,saveDir):
    #_imagList = re.findall('src="(.*?\.(jpg|png))',htmlCode)
    _imagList = re.findall('data-original="(.*?\.(jpg|png))',htmlCode)
    print(_imagList)
    #图片序号
    i = 1
    now = datetime.datetime.now()
    dirName = saveDir + "Spider" + now.strftime("%Y_%m_%d")
    try:
        os.mkdir(dirName)
    finally:
        for imag in _imagList:
#             print("下载第%d张图片：\n%s\n格式：%s"%(i,imag[0],imag[1]))
            if imag[1]=='png':
                if i<10:
                    saveName = dirName +'/imag_00'+str(i)+'.png'
                elif i<99:
                    saveName = dirName +'/imag_0'+str(i)+'.png'
                else:
                    saveName = dirName +'/imag_'+str(i)+'.png'
            elif imag[1]=='jpg':
                if i<10:
                    saveName = dirName +'/imag_00'+str(i)+'.jpg'
                elif i<99:
                    saveName = dirName +'/imag_0'+str(i)+'.jpg'
                else:
                    saveName = dirName +'/imag_'+str(i)+'.jpg'
            else:
                print("图片格式错误")
            urllib.request.urlretrieve(imag[0],saveName)      
            i+=1

htmlurl="https://www.douyu.com/directory/all"
saveDir="E:/python/spider/download/"
htmlCode = getHtmlCode(htmlurl)
getImag(htmlCode,saveDir)


