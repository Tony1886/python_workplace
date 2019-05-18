#coding:utf-8
'''
Created on 2017��5��8��

@author: Administrator
'''
import datetime
from tkinter import * 
import tkinter.simpledialog as tdl
import math

# def get_clock_step(base_pntr,long_pntr):
#     pos=[]
#     for i in range(60):
#         pos.append((base_pntr[0]+long_pntr*math.cos(i*math.pi/30),
#                     base_pntr[0]+long_pntr*math.sin(i*math.pi/30)))
#     return pos[45:]+pos[45:]
#     
#     
# 
# class Pointer(self,c_pntr,long_pntr,cvns,scale=None,super_pntr=None,width=1,fill="black"):
#     #����˵��
# #     c_pntr:������������
# #     long_pntr:���볤
#     
#     def draw(self):
        
# def get_clock(canvas,):       


def draw_number(canv,num_cen_x,num_cen_y,num_size,num_color,num):
    #num_cenΪ��������
    #num_sizeΪ���ִ�С
    #num_colorΪ������ɫ
    #numΪ��ʾ������
    beta=0.9
    if num==2 or num==3 or num==4 or num==5 or num==6 or num==8 or num==9:
        canv.create_line(num_cen_x-num_size/2+1,num_cen_y,num_cen_x+num_size/2,num_cen_y,fill=num_color)#�м��
    if num==0 or num==2 or num==3 or num==5 or num==6 or num==8:
        canv.create_line(num_cen_x-num_size/2+1,num_cen_y+num_size,num_cen_x+num_size/2,num_cen_y+num_size,fill=num_color)#�º�
    if num==0 or num==2 or num==3 or num==5 or num==6 or num==7 or num==8 or num==9:
        canv.create_line(num_cen_x-num_size/2+1,num_cen_y-num_size*beta,num_cen_x+num_size/2,num_cen_y-num_size*beta,fill=num_color)#�Ϻ�
    if num==0 or num==2 or num==6 or num==8:
        canv.create_line(num_cen_x-num_size/2+1,num_cen_y+num_size,num_cen_x-num_size/2+1,num_cen_y,fill=num_color)#���º�
    if num==0 or num==1 or num==3 or num==4 or num==5 or num==6 or num==7 or num==8 or num==9:
        canv.create_line(num_cen_x+num_size/2,num_cen_y+num_size,num_cen_x+num_size/2,num_cen_y,fill=num_color)#���º�
    if num==0 or num==4 or num==5 or num==6 or num==8 or num==9:
        canv.create_line(num_cen_x-num_size/2+1,num_cen_y,num_cen_x-num_size/2+1,num_cen_y-num_size*beta,fill=num_color)#���Ϻ�
    if num==0 or num==1 or num==2 or num==3 or num==4 or num==7 or num==8 or num==9:
        canv.create_line(num_cen_x+num_size/2,num_cen_y,num_cen_x+num_size/2,num_cen_y-num_size*beta,fill=num_color)#���Ϻ�
   
root = Tk()
w = Canvas(
    root,
    width=200,
    height=200,
    background="white"
    )
w.pack()
# line = w.create_line(0,100,100,100,fill="black")
# rect = w.create_rectangle(0,100,100,100)
# print(type(line))
# print(line)
# print(type(rect))
# line.rotate(math.pi/2)

# draw_number(w,150,100,20,"black",9)
draw_number(w,80,100,20,"black",0)
mainloop()
# now =datetime.datetime.now()
# 
# print(now.hour)
# print(now.minute)
# print(now.second)