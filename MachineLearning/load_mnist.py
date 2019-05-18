# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 22:07:56 2018
读取手写字符集

@author: Tan Zhijie
"""

import numpy as np  
import struct  
import matplotlib.pyplot as plt
   
def loadImageSet():
    filename = "train-images.idx3-ubyte"
    print ("load image set",filename  )
    binfile= open(filename, 'rb')  
    buffers = binfile.read()  
   
    head = struct.unpack_from('>IIII' , buffers ,0)  
    print ("head,",head)  
   
    offset = struct.calcsize('>IIII')  
    imgNum = head[1]  
    width = head[2]  
    height = head[3]  
    #[60000]*28*28  
    bits = imgNum * width * height  
    bitsString = '>' + str(bits) + 'B' #like '>47040000B'  
   
    imgs = struct.unpack_from(bitsString,buffers,offset)  
   
    binfile.close()  
    imgs = np.reshape(imgs,[imgNum,1,width*height])  
    print ("load imgs finished" ) 
    return imgs  
   
def loadLabelSet():  
    filename = "train-labels.idx1-ubyte"
    print ("load label set",filename  )
    binfile = open(filename, 'rb')  
    buffers = binfile.read()  
   
    head = struct.unpack_from('>II' , buffers ,0)  
    print ("head,",head)  
    imgNum=head[1]  
   
    offset = struct.calcsize('>II')  
    numString = '>'+str(imgNum)+"B"  
    labels = struct.unpack_from(numString , buffers , offset)  
    binfile.close()  
    labels = np.reshape(labels,[imgNum,1])  
   
    print ('load label finished')  
    return labels  
   
if __name__=="__main__":  
   
    imgs = loadImageSet()  
    labels = loadLabelSet()  
    
    # 显示一张图片和label
    index = 2
    plt.figure()
    plt.imshow(np.reshape(imgs[index],(28,28)),cmap = 'gray')
    plt.title(str(labels[index]))
    plt.show()
    
    #imgs = loadImageSet("t10k-images.idx3-ubyte")  
    #labels = loadLabelSet("t10k-labels.idx1-ubyte")  
    
