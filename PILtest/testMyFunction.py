# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:16:51 2018
test myFunction
@author: Tan Zhijie
"""

from myFunction import Diff
from PIL import Image
import matplotlib.pyplot as plt 

import sklearn

im0 = Image.open('lena.jpg').convert("L")#读取图片，并转化为灰度
#im = np.array(im0) 
opt = Diff()
opt.Ly = 6e-3
opt.Lx = 6e-3
opt.lamb = 532e-9
opt.d = 30e-2

#(Uo,yo,xo) = SFFT(im0,opt)
#(Uo,yo,xo) = angDiff(im0,opt)
(Uo,yo,xo) = Diff.FRTDiff(im0,opt)
#(UU,yo,xo) = SFFT(Uo,opt,2)
#(UU,yo,xo) = angDiff(Uo,opt,2)
(UU,yo,xo) = Diff.FRTDiff(Uo,opt,2)
plt.subplot(121)
plt.imshow(abs(Uo))
plt.subplot(122)
plt.imshow(abs(UU))
plt.show()