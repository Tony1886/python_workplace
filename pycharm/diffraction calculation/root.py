# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 13:02
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : root.py
# @Software: PyCharm

from tkinter import *
from tkinter import filedialog
from PIL import  ImageTk, Image
import matplotlib
import numpy as np
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import myFunction as myF


matplotlib.use('TkAgg')

def chooseImage():
        File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
        im = Image.open(File).convert("L")
        imSize = 256,256
        im.thumbnail(imSize,Image.ANTIALIAS)
        global processImage
        processImage = np.array(im)
        filename = ImageTk.PhotoImage(im)
        canvasOri.image = filename  # <--- keep reference of your image
        canvasOri.create_image(0,0,anchor='nw',image=filename)

def drawPic():
    drawPic.f.clf()
    drawPic.a = drawPic.f.add_subplot(111)
    global processImage
    # print(np.shape(processImage))
    Ny, Nx = np.shape(processImage)
    lamb = float(lambEntry.get())*10**(-9)
    Ly = float(LyEntry.get())*10**(-3)
    Lx = float(LxEntry.get())*10**(-3)
    d = float(dEntry.get())*10**(-2)

    opt = myF.Diff(lamb, Lx, Ly, d)

    if methodCal.get()==1:
        U1, yo, xo = myF.Diff.SFFT(processImage, opt)
    elif methodCal.get()==2:
        U1, yo, xo = myF.Diff.angDiff(processImage, opt)
    elif methodCal.get()==3:
        U1, yo, xo = myF.Diff.FRTDiff(processImage, opt)

    I = pow(abs(U1),2)
    # 绘制图形
    drawPic.a.imshow(I) # 当未选择图片时候，这里会出现一个错误
    ## 设定坐标
    ax = drawPic.f.gca()
    ax.set_xticks(np.linspace(0, Nx - 1, 2))  # 坐标是0~255
    ax.set_yticks(np.linspace(0, Ny - 1, 2))
    ax.set_xticklabels((str(round(xo[0] * 10 ** 6, 2))+'um', str(round(xo[Nx - 1] * 10 ** 6, 2))+'um'))
    ax.set_yticklabels((str(round(yo[Ny - 1] * 10 ** 6, 2))+'um', str(round(yo[0] * 10 ** 6, 2))+'um'))
    drawPic.canvas.show()
    #


root = Tk()
root.wm_title("Diffraction Calculation")
# root.geometry("600x400+300+100")

frameTop = Frame(root)
imageButton = Button(frameTop,text='choose an image',command=chooseImage)
imageButton.grid(row=0,column=0)
canvasOri = Canvas(frameTop)
canvasOri.grid(row=1, column=0)

frameTop.pack()

frameBottom = Frame(root)
frameBottom.pack(side=BOTTOM)
#### 设定的frame #####
frameSet = Frame(frameBottom)
frameSet.grid(row=0,column=0)
# 波长设定
lambLabel = Label(frameSet, text='lambda').grid(row=0, column=0)
lambEntry = Entry(frameSet)
lambEntry.insert(10,'532')
lambEntry.grid(row=0,column=1)
lambUnitLabel = Label(frameSet, text='nm').grid(row=0,column=2)



# 物体尺寸设定（Y方向）
LyLabel = Label(frameSet, text='Ly').grid(row=1, column=0)
LyEntry = Entry(frameSet)
LyEntry.insert(10,'3')
LyEntry.grid(row=1,column=1)
LyUnitLabel = Label(frameSet, text='mm').grid(row=1,column=2)

# 物体尺寸设定(X方向)
LxLabel = Label(frameSet, text='Lx').grid(row=2, column=0)
LxEntry = Entry(frameSet)
LxEntry.insert(10,'3')
LxEntry.grid(row=2,column=1)
LxUnitLabel = Label(frameSet, text='mm').grid(row=2,column=2)

# 衍射距离设定
dLabel = Label(frameSet, text='d').grid(row=3, column=0)
dEntry = Entry(frameSet)
dEntry.insert(10,'30')
dEntry.grid(row=3,column=1)
dUnitLabel = Label(frameSet, text='cm').grid(row=3,column=2)

###########  选择衍射计算类型的Frame  ########
frameCal = Frame(frameBottom)
frameCal.grid(row=1,column=0)
methodCal = IntVar()
methodCal.set(1)
Radiobutton(frameCal,text = 'Single FFT',variable = methodCal,value = 1).pack()
Radiobutton(frameCal,text = 'Double FFT',variable = methodCal,value = 2).pack()
Radiobutton(frameCal,text = 'Fractional FT',variable = methodCal,value = 3).pack()

## 计算的按钮
calButton = Button(frameCal,width=8, height=2, text='Calculation', command=drawPic)
calButton.pack()

############# 展示结果的canvas #########
drawPic.f = Figure(figsize=(5, 4), dpi=100)
drawPic.canvas = FigureCanvasTkAgg(drawPic.f, master=frameBottom)
drawPic.canvas.show()
drawPic.canvas.get_tk_widget().grid(row=0,column=1,rowspan=2)

root.mainloop()