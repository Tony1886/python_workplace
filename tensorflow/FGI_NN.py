# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 16:14
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : FGI_NN.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
import sys
import pylab as pl
sys.path.append(r'F:\python\python_workplace\myFunc')
import myFunction as mf

lamb = 532e-9 # wavelength
d = 30e-2     # diffraction distance
Ls = 1e-3     # size of source
Ns = 100      # pixels on source plane
ls = Ls/Ns
Lo = 0.5e-3
No = 20
lo = Lo/No
Lr = 20e-3
Nr = 100
lr = Lr/Nr

xs = mf.myGrid(Ns)*ls
xo = mf.myGrid(No)*lo
xr = mf.myGrid(Nr)*lr

Hr = mf.myH(xr, xs, lamb, d)
H1 = mf.myH(xo, xs, lamb, d)
H2 = mf.myH(xr, xo, lamb, d)
K = 10000
Es = np.exp(1j*2*np.pi*np.random.rand(Ns,K))

testNum = 5000
o = np.random.rand(No,1)

Er = np.matmul(Hr, Es)
Ir = np.mat(pow(abs(Er),2))
Eo = np.matmul(H1, Es)
Et = np.matmul(H2,np.matmul(np.diag(o.flat),Eo))
It = np.mat(pow(abs(Et),2))

point_x = int(Nr/2)
GI = K*Ir*np.transpose(np.mat(It[point_x, :]))/(np.sum(Ir,1)*np.sum(It[point_x, :]))
pl.plot(GI)
pl.show()

print('end')

