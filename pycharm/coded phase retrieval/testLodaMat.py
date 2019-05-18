# -*- coding: utf-8 -*-
# @Time    : 2019/2/2 14:06
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : testLodaMat.py
# @Software: PyCharm

from tkinter import *
from tkinter import filedialog
import numpy as np
import scipy.io as scio

def chooseData():
    File = filedialog.askopenfilename(parent=root, initialdir="D:\python\python_workplace\pycharm\coded phase retrieval", title='Choose an image.')
    data = scio.loadmat(File)
    global diffData,masks
    diffData = data['diffData']
    masks = data['masks']


def test():
    global diffData
    diffSize = np.shape(diffData)
    print(diffSize[-1])


root = Tk()
root.wm_title('Phase Retrieval')

loadButton = Button(root, text = 'choose data', command = chooseData)
loadButton.pack()

testButton = Button(root,text = 'test', command = test)
testButton.pack()

root.mainloop()