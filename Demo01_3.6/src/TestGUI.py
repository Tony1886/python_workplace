#coding:utf-8
#һ�������ֵ�С��Ϸ
from tkinter import *
import tkinter.simpledialog as dl
import tkinter.messagebox as mb

#tkinter GUI input output example
#setting gui
root = Tk()
w = Label(root,text="Guess Number Game")
w.pack()#�Զ����ڴ�С

#��������
number = 55

while True:

    guess = dl.askinteger("number",'Enter a number')
    if guess<number:
        #setting message box
        mb.showinfo("Hint",'little')
    elif guess>number:
        mb.showinfo("Hint",'bigger')
    else:
        mb.showinfo("Hint",'bingo')
        break
    

