# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:21:04 2018

@author: Tan Zhijie
"""

import tkinter as tk

root = tk.Tk()
root.title('my window')
root.geometry("200x100+100+200")



on_hit = False
def hit_me():
    global on_hit
    if on_hit ==False:
        on_hit = True
        var.set("you hit me")
    else:
        on_hit = False
        var.set("")   
        
        
var = tk.StringVar()
label = tk.Label(root,textvariable = var,bg='green',font = ('Arial',12),width = 15,height = 2)
button = tk.Button(root,text = 'hit me',width = 15,height=2,command = hit_me)
label.pack()
button.pack()

root.mainloop()