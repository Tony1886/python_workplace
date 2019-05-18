# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:47:42 2018

@author: Tan Zhijie
"""

import tkinter as tk

root = tk.Tk()
root.title('my window')
root.geometry("200x200")

var1 = tk.StringVar()
var2 = tk.StringVar()
l = tk.Label(root,bg='yellow',width=4,textvariable = var1)
l.pack()
def printSelect():
    value = lb.get(lb.curselection())
    var1.set(value)

b1 = tk.Button(root,text = 'print selection',width = 15,height=2,command = printSelect)
b1.pack()
var2.set((11,22,33,44))
lb = tk.Listbox(root,listvariable = var2)
list_item = [1,2,3,4]
for item in list_item:
    lb.insert('end',item)
lb.pack()


root.mainloop()