# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:19:56 2018

@author: Tan Zhijie
"""

from PIL import Image
import glob,os
import os
import tkinter.filedialog 

default_dir = r"C:\Users\lenovo\Desktop"  # 设置默认打开目录
fname = tkinter.filedialog.askopenfilename(title=u"选择文件",
                                     initialdir=(os.path.expanduser(default_dir)))

print (fname)  # 返回文件全路径
print (tkinter.filedialog.askdirectory())  # 返回目录路径



size = 128,128
for infile in glob.glob("*.jpg"): # glob的作用是文件搜索，返回的是一个列表
    file,ext = os.path.splitext(infile) # 将文件的文件名和拓展名分开，用于之后的保存重命名
    im = Image.open(infile)
    im.thumbnail(size,Image.ANTIALIAS)  # 等比例缩放
    #im.save(file+".thumbnail","JPEG")
    im.show() # 显示缩略图
    print (im.size,im.mode)