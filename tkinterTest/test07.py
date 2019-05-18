# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:23:28 2018

@author: Tan Zhijie
"""
import tkinter as tk


root = tk.Tk()
root.title('my window')
root.geometry("200x200")

canvas = tk.Canvas(root,bg = 'blue',height = 100, width = 200)
#image_fil = tk.PhotoImage(file = '')
#image = canvas.create_image(10,10,anchor = 'nw',image=image_file)
x0,y0,x1,y1 = 50,50,80,80
line = canvas.create_line(x0,y0,x1,y1)
oval = canvas.create_oval(x0,y0,x1,y1,fill='red')
arc = canvas.create_arc(x0+30,y0+30,x1+30,y1+30,start = 0,extent = 180)
rect = canvas.create_rectangle(100,30,100+30,30+30)
canvas.pack()

def moveit():
    canvas.move(rect,0,2)
     
b = tk.Button(root,text = 'move',command = moveit).pack()


root.mainloop()