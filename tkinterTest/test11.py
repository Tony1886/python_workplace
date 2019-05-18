# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:32:12 2018

@author: Tan Zhijie
"""

import tkinter as tk

root = tk.Tk()
root.title('my window')
root.geometry("200x200")

'''
tk.Label(root,text = '1').pack(side='top')
tk.Label(root,text = '1').pack(side='bottom')
tk.Label(root,text = '1').pack(side='left')
tk.Label(root,text = '1').pack(side='right')
'''

'''
for i in range(4):
    for j in range(3):
        tk.Label(root,text = i+j).grid(row = i,column = j,ipadx = 10,ipady =10)
'''

tk.Label(root,text = 1).place(x=10,y=100,anchor='nw')

root.mainloop()