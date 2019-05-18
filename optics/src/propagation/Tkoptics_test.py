#coding:utf-8
'''
Created on 2017年5月21日

@author: Administrator
'''

from tkinter import * 
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from matplotlib.figure import Figure


# matplotlib.use('TKAgg')

def figDraw():    
    global maxL
    x=linspace(-maxL, maxL, 500)
    y=sin(x) 
    fig.clf()#删除画布，以防重合
    a=fig.add_subplot(121)
    b=fig.add_subplot(122)  
    a.plot(x,y)#绘制一张图
    [X,Y]=meshgrid(x,y)
    F=sqrt(pow(X,2)+pow(Y,2))
    b.imshow(F)#imshow画二维图
    canvas = FigureCanvasTkAgg(fig, master=root)#将图片放入画布
    canvas.get_tk_widget().grid(row=1,column=0)#将图片放入TKinter中
    canvas.draw()
def setNum():
    global maxL
    maxL=maxL+pi
    figDraw()
#     print(maxL)
    label['text']=str(int(maxL/pi))+'pi'
    
root = Tk()
global maxL
maxL=pi
fig=Figure(figsize=(5,4),dpi=100)#构造一张图
a=fig.add_subplot(121)
b=fig.add_subplot(122)#给图加两个子图
canvas = FigureCanvasTkAgg(fig, master=root)#将图片放入画布
canvas.get_tk_widget().grid(row=1,column=0)#将图片放入TKinter中


topFrame = Frame(root)
button1 = Button(topFrame,text = 'test',command=figDraw)
button1.pack(fill = X,side = LEFT,expand=YES)
button2 = Button(topFrame,text = 'change range',command = setNum)
button2.pack(side=LEFT,padx=5)
label = Label(text='pi')
label.grid(row=2,column=0,sticky=W)

topFrame.grid(row=0,column=0,sticky=W)

# topFrame.pack(side=TOP,fill=X)
# canvas.get_tk_widget().pack(side=LEFT,fill=X)
# label.pack(side=BOTTOM,fill=X)

mainloop()