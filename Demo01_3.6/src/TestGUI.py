#coding:utf-8
#一个猜数字的小游戏
from tkinter import *
import tkinter.simpledialog as dl
import tkinter.messagebox as mb

#tkinter GUI input output example
#setting gui
root = Tk()
w = Label(root,text="Guess Number Game")
w.pack()#自动调节大小

#设置数字
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
    

