# coding：utf-8
# @Created on 2018年1月26日 下午2:39:53
# @author: Administrator


from math import *
from numpy import *
from matplotlib import *
import matplotlib.pyplot as plt
from numpy.dual import fft, fft2, ifft,ifft2
from numpy.fft.helper import *
from numpy.core.function_base import linspace


def mfft(f):
    if size(f,0)==1 or size(f,1)==1:
        F = fftshift(fft(ifftshift(f)))
    else:
        F = fftshift(fft2(ifftshift(f)))
    return F

def mifft(f):
    if size(f,0)==1 or size(f,1)==1:
        F = fftshift(ifft(ifftshift(f)))
    else:
        F = fftshift(ifft2(ifftshift(f)))
    return F