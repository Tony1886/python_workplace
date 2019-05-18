# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:20:28 2018

@author: Tan Zhijie
"""

import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.title('my window')
root.geometry("200x200")

def hit_me():
    #tk.messagebox.showinfo(title='Hi',message='hahahaha')
    #tk.messagebox.showwarning(title='Hi',message='nonono')
    #tk.messagebox.showerror(title='Hi',message='No!never')
    #print(tk.messagebox.askquestion(title='Hi',message = 'hahaha')) # return 'yes' 'no'
    print(tk.messagebox.askretrycancel(title='Hi',message = 'hahaha')) # return 'True' 'False'
    
b = tk.Button(root,text = 'hit me',command = hit_me).pack()

root.mainloop()