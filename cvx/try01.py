# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:25:26 2018

@author: Tony
"""

import cvxpy as cvx

l = cvx.Variable()    # 波带片宽度
d1 = cvx.Variable()   # 光源到波带片的距离
L = cvx.Variable()    # 波带片的距离

r = 100               # 波带片焦距与宽度的比值
h = 200e-6            # 初始光源大小
s = 50e-2             # 限孔到散射屏的距离

f = l/r               # 波带片焦距
d2 = 1/(1/f-1/d1)     # 成像距离--波带片到像面的距离，限孔的位置
a = d2/d1             # 成像放大率
hi = a*h              # 像的大小-- 限孔的大小 

a_x = d2              
a_y = (L-h/2)/d1*d2+L # the coordinate of the point a
b_x = d2
b_y = (L+h/2)/d1*d2+L # 
c_x = 0
c_y = L+l/2           #
d_x = 0
d_y = L-l/2           #

kac = (a_y-c_y)/(a_x-c_x)
kbd = (b_y-d_y)/(b_x-d_x)

yl = kac*(d2+s)+c_y   # 照射散射屏下面的位置
yh = kbd*(d2+s)+d_y   # 照射散射屏上面的位置

l.vale = 2e-3
d1.value = 1
L.value = 4e-3;
#yl.value = 0.5e-2
print(yh.value)


