#coding:utf-8
'''
Created on 2017年6月5日

@author: Administrator
'''

from numpy import *

f = open('level02.txt','rb')
newFile = open('New_level02.txt','w')
content = f.read(1).decode('utf-8')

while len(content)>0:
    if ord(content) >=ord('a') and ord(content)<=ord('z'):
        newcontent = ord(content)+2-(ord('z')-ord('a')+1)*int((ord(content)+1)/(ord('z')))
    else:
        newcontent = ord(content)
    content = f.read(1).decode('utf-8')   
    newFile.write(chr(newcontent))

f.close()
newFile.close()
# print(content)