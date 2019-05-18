# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:34:52 2018

@author: Tan Zhijie
"""

import tkinter as tk

root = tk.Tk()
root.title('my window')
root.geometry("200x200")

e = tk.Entry(root,show = None)
e.pack()
def insertPoint():
    var  = e.get()
    t.insert("insert",var)
def insertEnd():
    var = e.get()
    #t.insert("end",var)
    t.insert(1.1,var)

        
b1 = tk.Button(root,text = 'insert point',width = 15,height=2,command = insertPoint)
b1.pack()

b1 = tk.Button(root,text = 'insert end',width = 15,height=2,command = insertEnd)
b1.pack()

t = tk.Text(root,height = 2)
t.pack()

root.mainloop()