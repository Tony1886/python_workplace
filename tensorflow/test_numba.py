# -*- coding: utf-8 -*-
# @Time    : 2018/8/31 14:50
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : test_numba.py
# @Software: PyCharm

# test numba using svd

from numba import jit
import time
from numpy import linalg
from timeit import timeit
from numpy import arange

# from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
from matplotlib.pylab import imshow, jet, show, ion
import numpy as np

from numba import jit


@jit
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return 255

@jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image
image = np.zeros((500 * 2, 750 * 2), dtype=np.uint8)
s = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
e = timer()
print(e - s)
imshow(image)
# jet()
# ion()
show()

'''
do not work
@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

start_time = time.time()
M = 500
N = 500
a = arange(M*N).reshape(M,N)
print(sum2d(a))

end_time = time.time()
t = end_time - start_time
print(t)
'''

'''
并不能看出差异
@jit
def func():
    M = 1000
    N = 1000
    X = np.random.rand(M,N)
    linalg.svd(X)


# timeit(函数名_字符串，运行环境_字符串，number=运行次数)
t = timeit(stmt = 'func()', setup = 'from __main__ import func', number=100)
print(t)
'''

