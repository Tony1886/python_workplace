# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:57:00 2018

@author: Tan Zhijie
"""

import tkinter as tk

root = tk.Tk()
root.title('my window')
root.geometry("200x200")

var1 = tk.StringVar()

l = tk.Label(root,bg='yellow',width=20,text = 'empty')
l.pack()

def print_selection():
    l.config(text='you have selected ' + var1.get())
r1 = tk.Radiobutton(root,text = 'Option A',variable = var1, value = 'A',command = print_selection)
r2 = tk.Radiobutton(root,text = 'Option B',variable = var1, value = 'B',command = print_selection)
r3 = tk.Radiobutton(root,text = 'Option C',variable = var1, value = 'C',command = print_selection)
r1.pack()
r2.pack()
r3.pack()

root.mainloop()