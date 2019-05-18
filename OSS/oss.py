# -*- coding: utf-8 -*-
# @Time    : 2018/10/30 14:12
# @Author  : Tan Zhijie
# @Email   : tanzj@siom.ac.cn
# @File    : oss.py
# @Software: PyCharm

# coding: utf-8

# # implementação do HIO puro
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.fftpack import fftn, ifftn

# In[13]:


# Implementando o OSS puro

iterations = 10000
beta = 0.9
filtercount = 10.  # iterations/100

# Assign random phases


imagSize = np.shape(DifPad);

R2D = np.zeros((imagSize[0], imagSize[1], int(filtercount))).astype(np.float32)
toperrs = np.ones((filtercount)).astype(np.float32)
step = np.array(range(np.int(iterations / filtercount), np.int(iterations + 1), np.int(iterations / filtercount)))
filtnum = 1
HIOfirst = 1.
kfilter = np.zeros((imagSize[0], imagSize[1])).astype(np.float32)
x = np.arange(-imagSize[1] / 2, imagSize[1] / 2, 1)
y = np.arange(-imagSize[0] / 2, imagSize[0] / 2, 1)
xx, yy = np.meshgrid(x, y, sparse=True)

X = np.array(range(1, iterations + 1))
sigma = (filtercount + 1 - np.ceil(X * filtercount / iterations)) * np.ceil(iterations / filtercount)
sigma = ((sigma - np.ceil(iterations / filtercount)) * (2 * imagSize[0]) / np.max(sigma)) + (2 * imagSize[0] / 10)

np.random.seed(1)  # definning randon seed for randon phase values to be the same betwwen reconstructions
phase_angle = np.random.rand(imagSize[0], imagSize[1]).astype(np.float32);
phase_angle = phase_angle * 2 * np.pi

# Define initial k, r space
initial_k = np.fft.ifftshift(DifPad)
ampKSpace = np.fft.ifftshift(DifPad)
initial_k[ampKSpace == -1] = 0
k_space = initial_k * np.exp(1j * phase_angle);

buffer_r_space = scipy.real(ifftn(k_space))
r_space = scipy.real(ifftn(k_space))

# Preallocate error arrays
RfacF = np.zeros((iterations, 1)).astype(np.float32);
counter1 = 0;
errorF = 1;

# HIO iterations
iter = 1.
while (iter <= iterations):
    #    if iter == 300:
    #        break

    # HIO with Support & Positivity constraint
    r_space = np.real(np.fft.ifftn(k_space));

    sample = r_space * Mask;

    r_space = buffer_r_space - beta * r_space
    sample[sample < 0] = r_space[sample < 0];

    # r_space[Mask==1] = sample[Mask==1];

    # OSS filter
    if (HIOfirst == 0 or iter > np.ceil(
            iterations / filtercount)):  # and iter < np.ceil(iterations-iterations/filtercount):
        kfilter = np.exp(-(((np.sqrt((yy) ** 2 + (xx) ** 2) ** 2)) / (2 * sigma[int(iter - 1)] ** 2)))  # gaussian
        # with the matrixes from a while back
        kfilter = (kfilter / np.max(kfilter)).astype(np.float32)  # to make sure it is normalized

        ktemp = scipy.fftpack.fftshift(scipy.fftpack.fft2(r_space))
        ktemp = ktemp * kfilter  # convolution
        r_space = scipy.real(scipy.fftpack.ifft2(scipy.fftpack.fftshift(ktemp)))  # after gaussian
        # print(iter)
        # filter has been put on

    #    if iter < np.ceil(iterations-iterations/filtercount):
    #        beta = 1.

    if np.mod((iter - 1), iterations / filtercount) == 0:
        r_space = R2D[:, :, filtnum - 1]

    else:
        r_space[Mask == 1] = sample[Mask == 1]  # Makes it unblurry

    buffer_r_space = r_space;

    # k_space = np.fft.fftn(r_space);
    k_space = fftn(r_space)

    phase_angle = np.angle(k_space);
    k_space[ampKSpace != -1] = ampKSpace[ampKSpace != -1] * np.exp(
        1j * phase_angle[ampKSpace != -1]);  # funciona também, mas na forma de baixo é pra ser mais universal

    # notthe_angle = np.divide(k_space,np.absolute(k_space))
    # notthe_angle[np.absolute(k_space)==0] = 1

    # noBeamStop = np.logical_or(np.absolute(k_space)==0,ampKSpace!=-1)
    # k_space[noBeamStop] = ampKSpace[noBeamStop] * notthe_angle[noBeamStop]

    # Calculate errors
    # Calculate error in reciprocal space
    Ktemp = r_space * Mask
    # Ktemp = np.absolute(np.fft.fftshift(np.fft.fft2(Ktemp)));
    Ktemp = np.absolute(fftn(Ktemp))
    errorF = np.sum((np.absolute(Ktemp[ampKSpace != -1] - ampKSpace[ampKSpace != -1]))) / np.sum(
        ampKSpace[ampKSpace != -1]);
    RfacF[counter1] = errorF;

    # Determine interations with best error
    filtnum = int(np.ceil((iter) * filtercount / iterations));

    if errorF <= toperrs[filtnum - 1]:
        toperrs[filtnum - 1] = errorF
        R2D[:, :, filtnum - 1] = r_space

    counter1 += 1;

    iter = iter + 1

r_space = R2D[:, :, -1]
errorF = toperrs[-1]
RfacF[-1] = errorF

plt.subplot(2, 2, 1)
imgplot = plt.imshow(np.log10(np.absolute(DifPad)))
plt.subplot(2, 2, 2)
imgplot = plt.imshow(np.log10(np.absolute(np.fft.fftshift(k_space))))
plt.subplot(2, 2, 3)
imgplot = plt.imshow(r_space * Mask)
plt.subplot(2, 2, 4)
plt.plot(RfacF)
plt.ylabel('Fourier R-factor')
plt.xlabel('Iteration')
plt.title('Final R-factor: %s' % (errorF * 100))
plt.tight_layout()
plt.suptitle('Reconstruction using OSS')

# while True:
#    plt.pause(0.5)