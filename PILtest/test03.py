# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 22:11:56 2018

@author: tanzj
"""

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk




def printcoords():
        File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
        im = Image.open(File)
        imSize = 128,128
        im.thumbnail(imSize,Image.ANTIALIAS)
        filename = ImageTk.PhotoImage(im)
        
        canvas.image = filename  # <--- keep reference of your image
        
        canvas.create_image(0,0,anchor='nw',image=filename)
 
root=Tk()
root.wm_title("Diffraction Calculation")
root.geometry("600x400+300+100")
       
#setting up a tkinter canvas with scrollbars
frame = Frame(root, bd=2, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
xscroll = Scrollbar(frame, orient=HORIZONTAL)
xscroll.grid(row=1, column=0, sticky=E+W)
yscroll = Scrollbar(frame)
yscroll.grid(row=0, column=1, sticky=N+S)
canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
xscroll.config(command=canvas.xview)
yscroll.config(command=canvas.yview)
frame.pack(fill=BOTH, expand=0)

# a button for choose an image
Button(root,text='image', command=printcoords).pack()


root.mainloop()