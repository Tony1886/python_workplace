# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:03:44 2018

@author: Tan Zhijie
"""

import tkinter as tk

root = tk.Tk()
root.title('my window')
root.geometry("200x200")


l = tk.Label(root,bg='yellow',width=20,text = 'empty')
l.pack()

def print_selection(m):
    l.config(text='you have selected ' + m)

s = tk.Scale(root,label = 'try me',from_ = 5,to = 11,orient = tk.HORIZONTAL , length = 200,
             showvalue=0,tickinterval = 3,resolution = 0.01,command = print_selection)
s.pack()

root.mainloop()