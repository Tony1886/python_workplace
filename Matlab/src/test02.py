#coding:utf-8
'''
Created on 2017年5月25日

@author: Administrator
'''
from numpy import *

content = 'u'
newcontent1 = ord(content)+2-(ord('z')-ord('a')+1)*int((ord(content)+1)/ord('z'))
newcontent2 = ord(content)+2
print(chr(newcontent1))
print(chr(newcontent2))
print(ord(content))
print(ord('z'))
# print(int(2/3))