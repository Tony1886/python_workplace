# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 13:53
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : try02.py
# @Software: PyCharm

# -*- coding: UTF-8 -*-
#python tkinter image
from tkinter import *

master = Tk()


root = Frame(master)
lambLabel = Label(root, text='lambda').grid(row=0, column=0)
lambEntry = Entry(root).grid(row=0,column=1)
lambUnitLabel = Label(root, text='nm').grid(row=0,column=2)
# 物体尺寸设定（Y方向）
LyLabel = Label(root, text='Ly').grid(row=1, column=0)
LyEntry = Entry(root).grid(row=1,column=1)
LyUnitLabel = Label(root, text='mm').grid(row=1,column=2)
# 物体尺寸设定(X方向)
LxLabel = Label(root, text='Lx').grid(row=2, column=0)
LxEntry = Entry(root).grid(row=2,column=1)
LxUnitLabel = Label(root, text='mm').grid(row=2,column=2)
# 衍射距离设定
dLabel = Label(root, text='d').grid(row=3, column=0)
dEntry = Entry(root).grid(row=3,column=1)
dUnitLabel = Label(root, text='cm').grid(row=3,column=2)
#### 展示图 ##########

root.pack()

fram = Frame(master)
fram.pack()
root.mainloop()