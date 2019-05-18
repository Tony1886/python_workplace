# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 12:57
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : diffractionTest.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy.fft import fft2,ifft2,fftshift,ifftshift
import sys
sys.path.append('D:\python\python_workplace\PILtest')
import myFunction as myF


im0 = Image.open('lena.jpg').convert("L")#读取图片，并转化为灰度
im = np.array(im0)

#设置物面大小
Ly = 6e-3
Lx = 6e-3
Ny, Nx = np.shape(im)
lx = Lx/Nx
ly = Ly/Ny
xi = np.linspace(-Lx/2, Lx/2, Nx)
yi = np.linspace(-Ly/2, Ly/2, Ny)
[Yi, Xi] = np.meshgrid(yi, xi)
# 光路设计
lamb = 532e-9
d = 30e-2

Lo_y = Ny*lamb*d/Ly
Lo_x = Nx*lamb*d/Lx
xo = np.linspace(-Lo_x/2, Lo_x/2, Nx)
yo = np.linspace(-Lo_y/2, Lo_y/2, Ny)
[Yo, Xo] = np.meshgrid(yo,xo) # 得到观察面的坐标
Fresnel = np.exp(1j*np.pi/lamb/d*(pow(Xi,2)+pow(Yi,2)))
phase = np.exp(1j*2*np.pi/lamb*d)/(1j*lamb*d)*np.exp(1j*np.pi/lamb/d*(pow(Xo, 2)+pow(Yo, 2)))

Uo = np.multiply(fftshift(fft2(ifftshift(np.multiply(im,Fresnel)))),phase)
Io = pow(abs(Uo), 2)

opt = myF.Diff(lamb, Lx, Ly, d)
U1, yo, xo = myF.Diff.SFFT(im, opt)
U2, yi, xi = myF.Diff.SFFT(U1, opt, 2)
I1 = pow(abs(U1), 2)
I2 = pow(abs(U2), 2)

plt.imshow(I1)
ax = plt.gca()
ax.set_xticks(np.linspace(0, Nx-1, 2))# 坐标是0~255
ax.set_yticks(np.linspace(0, Ny-1, 2))
ax.set_xticklabels((str(round(xo[0]*10**6, 2)), str(round(xo[Nx-1]*10**6, 2))))
ax.set_yticklabels((str(round(yo[Ny-1]*10**6, 2)), str(round(yo[0]*10**6, 2))))

plt.show()