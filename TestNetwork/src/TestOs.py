#coding:utf-8
'''
Created on 2017��5��11��
beautifulsoup 测试
@author: Administrator
'''
import urllib.request
from bs4 import BeautifulSoup

html = urllib.request.urlopen("http://www.baidu.com")
soup = BeautifulSoup(html)
print(type(soup))
print(soup.title)#打印网页标题
print(soup.head)
print(soup.a)