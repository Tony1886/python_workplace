# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:11:29 2018

@author: Tan Zhijie
"""

import tkinter as tk

root = tk.Tk()
root.title('my window')
root.geometry("200x200")


l = tk.Label(root,bg='yellow',width=20,text = 'empty')
l.pack()

def print_selection():
    #global var1,var2
    if var1.get()==1 and var2.get()==1:
        l.config(text='love both ' )
    elif var1.get()==1 and var2.get()==0:
        l.config(text='love python ' )
    elif var1.get()==0 and var2.get()==1:
        l.config(text='love C++ ' )
    else:
        l.config(text='love neither ' )
    

var1 = tk.IntVar()
var2 = tk.IntVar()
c1 = tk.Checkbutton(root,text = 'python',variable = var1, onvalue = 1,offvalue =0,
                    command = print_selection)
c2 = tk.Checkbutton(root,text = 'C++',variable = var2, onvalue = 1,offvalue =0,
                    command = print_selection)

c1.pack()
c2.pack()
root.mainloop()