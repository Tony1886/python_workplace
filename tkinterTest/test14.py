# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:57:36 2018

@author: Tan Zhijie
"""

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

if __name__ == "__main__":
    root = Tk()

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
    frame.pack(fill=BOTH,expand=1)


    #function to be called when mouse is clicked
    def printcoords():
        File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
        filename = ImageTk.PhotoImage(Image.open(File))
        canvas.image = filename  # <--- keep reference of your image
        canvas.create_image(0,0,anchor='nw',image=filename)

    Button(root,text='choose',command=printcoords).pack()
    root.mainloop()