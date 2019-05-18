# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:01:53 2018

@author: Tan Zhijie
"""

import tkinter as tk


root = tk.Tk()
root.title('my window')
root.geometry("200x200")

l = tk.Label(root,bg='yellow',width=20,text = 'empty')
l.pack()

counter = 0
def do_job():
    global counter
    l.config(text = 'do'+str(counter))
    counter+=1
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label='File',menu = filemenu)
filemenu.add_command(label = 'New',command = do_job)
filemenu.add_command(label = 'Open',command = do_job)
filemenu.add_command(label = 'Save',command = do_job)
filemenu.add_separator()
filemenu.add_command(label = 'Exit',command = do_job)

submenu = tk.Menu(filemenu)

root.config(menu = menubar)
root.mainloop()