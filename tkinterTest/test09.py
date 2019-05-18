# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:15:28 2018

@author: Tan Zhijie
"""

import tkinter as tk

root = tk.Tk()
root.title('my window')
root.geometry("200x200")

tk.Label(root,text='on the window').pack()
frm = tk.Frame(root)
frm.pack()

frm_l = tk.Frame(frm)
frm_r = tk.Frame(frm)
frm_l.pack(side='left')
frm_r.pack(side='right')

tk.Label(frm_l,text='on the frm_l1').pack()
tk.Label(frm_l,text='on the frm_l2').pack()
tk.Label(frm_r,text='on the frm_r').pack()

root.mainloop()