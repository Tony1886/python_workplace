# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 9:46
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : main.py
# @Software: PyCharm

from tkinter import *
from tkinter import filedialog
from PIL import  ImageTk, Image
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy.fft import fft,ifft,fft2,ifft2,fftshift,ifftshift
from numpy import pi,exp
import math

# matplotlib.use('TkAgg')


def mfft(f):
    f = np.array(f)
    F = np.zeros(np.shape(f))
    if np.size(f, 0) == 1 or np.size(f, 1) == 1:
        F = fftshift(fft(ifftshift(f)))
    else:
        F = fftshift(fft2(ifftshift(f)))
    return F


# 快速逆傅里叶变换，调整0频
def mifft(f):
    f = np.array(f)
    F = np.zeros(np.shape(f))
    if np.size(f, 0) == 1 or np.size(f, 1) == 1:
        F = fftshift(ifft(ifftshift(f)))
    else:
        F = fftshift(ifft2(ifftshift(f)))
    return F


# 划分网格
def myGrid(N):
    if np.size(N) == 1:
        pass
        if np.mod(N, 2) == 0:
            x = np.linspace(-N / 2, N / 2 - 1, N)
        else:
            x = np.linspace(-(N - 1) / 2, (N - 1) / 2, N)
        return x
    else:
        Ny = N[0]
        Nx = N[1]
        x = myGrid(Nx)
        y = myGrid(Ny)
        [Y, X] = np.meshgrid(y, x)
    return (Y, X)


# 构造光场传播的类
class Diff:
    #    类初初始化，得到4个参数
    def __init__(self, lamb=532e-9, Lx=1e-3, Ly=1e-3, d=30e-2):
        self.lamb = lamb  # 波长
        self.Lx = Lx
        self.Ly = Ly  # 尺寸
        self.d = d  # 衍射距离

    # 衍射积分的SFFT方法
    def SFFT(Ui, opt, mode=1):
        lamb = opt.lamb
        Lx = opt.Lx
        Ly = opt.Ly
        d = opt.d

        Ui = np.array(Ui)
        #    Ny,Nx = myGrid(np.shape(Ui))
        Ny, Nx = np.shape(Ui)
        Lox = Nx * lamb * d / Lx
        Loy = Ny * lamb * d / Ly  # 衍射面大小
        lox = Lox / Nx
        loy = Loy / Ny  # 衍射面分辨率

        xi = myGrid(Nx) * Lx / Nx
        yi = myGrid(Ny) * Ly / Ny

        xo = myGrid(Nx) * lox
        yo = myGrid(Ny) * loy

        [Yi, Xi] = np.meshgrid(yi, xi)
        [Yo, Xo] = np.meshgrid(yo, xo)
        if mode == 1:
            Fresnel = exp(1j * pi / lamb / d * (pow(Xi, 2) + pow(Yi, 2)))
            phase = exp(1j * 2 * pi / lamb * d) / (1j * lamb * d) * exp(1j * pi / lamb / d * (pow(Xo, 2) + pow(Yo, 2)))
            Uo = mfft(np.multiply(Ui, Fresnel))
            Uo = np.multiply(Uo, phase)
            return (Uo, yo, xo)
        else:
            back_Fresnel = exp(-1j * pi / lamb / d * (pow(Xi, 2) + pow(Yi, 2)))
            back_phase = exp(-1j * 2 * pi / lamb * d) / (1j * lamb * d) * exp(
                -1j * pi / lamb / d * (pow(Xo, 2) + pow(Yo, 2)))
            Uo = mifft(np.multiply(Ui, back_phase))
            Uo = np.multiply(Uo, back_Fresnel)
            return (Uo, yi, xi)


    # 衍射积分的角谱衍射方法，即一种DFFT方法
    def angDiff(Ui, opt, mode=1):
        lamb = opt.lamb
        Lx = opt.Lx
        Ly = opt.Ly
        d = opt.d
        Ui = np.array(Ui)

        Ny, Nx = np.shape(Ui)

        # 物面尺度
        xi = myGrid(Nx) * Lx / Nx
        yi = myGrid(Ny) * Ly / Ny

        # 物体面频率信息
        xf = myGrid(Nx) * 1 / Lx
        yf = myGrid(Ny) * 1 / Ly
        [Yf, Xf] = np.meshgrid(yf, xf)

        H = exp(1j * 2 * pi / lamb * d * np.sqrt(1 - pow(lamb * Xf, 2) - pow(lamb * Yf, 2)));
        if mode == 1:
            Uf = mfft(Ui)
            Uo = mifft(np.multiply(Uf, H))
            return (Uo, yi, xi)
        else:
            Uo = mifft(np.multiply(mfft(Ui), np.conj(H)))
            return (Uo, yi, xi)

    # 衍射积分的分数傅里叶变换方法
    def FRTDiff(Ui, opt, mode=1):
        lamb = opt.lamb
        Lx = opt.Lx
        Ly = opt.Ly
        d = opt.d
        Ui = np.array(Ui)

        Ny, Nx = np.shape(Ui)
        # 物面坐标
        yi = myGrid(Ny) * Ly / Ny
        xi = myGrid(Nx) * Lx / Nx
        [Yi, Xi] = np.meshgrid(yi, xi)
        # 物面频谱坐标
        xf = myGrid(Nx) * 1 / Lx
        yf = myGrid(Ny) * 1 / Ly
        [Yf, Xf] = np.meshgrid(yf, xf)

        f1 = Lx ** 2 / lamb / Nx
        sita = math.atan(d / f1)
        beta = math.cos(sita)
        p = sita / 2 / pi

        phase = 1j * exp(1j * pi * (1 - p) / 2) * exp(
            -1j * pi / lamb / f1 * (pow(Xi, 2) + pow(Yi, 2)) * math.tan(sita / 2))
        FRT1 = exp(-1j * pi / lamb / f1 * (pow(Xi, 2) + pow(Yi, 2)) * math.tan(sita / 2))
        FRT2 = exp(-1j * pi * lamb * f1 * math.sin(sita) * (pow(Xf, 2) + pow(Yf, 2)))

        Lo = Lx / beta
        xo = myGrid(Nx) * Lo / Nx
        yo = myGrid(Ny) * Lo / Ny
        if mode == 1:
            Uo = np.multiply(mifft(np.multiply(mfft(np.multiply(Ui, FRT1)), FRT2)), phase)
            return (Uo, yo, xo)
        else:
            Uo = np.multiply(mifft(np.multiply(mfft(np.multiply(Ui, np.conj(phase))), np.conj(FRT2))), np.conj(FRT1))
            return (Uo, yi, xi)

## 打开一张图片
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

## 计算传播并显示结果
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

    opt = Diff(lamb, Lx, Ly, d)

    if methodCal.get()==1:
        U1, yo, xo = Diff.SFFT(processImage, opt)
    elif methodCal.get()==2:
        U1, yo, xo = Diff.angDiff(processImage, opt)
    elif methodCal.get()==3:
        U1, yo, xo = Diff.FRTDiff(processImage, opt)

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


## main ###
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