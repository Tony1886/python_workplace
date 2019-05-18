# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:39:03 2018
一个小的登陆程序
@author: Tan Zhijie
"""

import tkinter as tk


window = tk.Tk()
window.title('Welcome to My world')
window.geometry("450x300")
tk.Label(window,text='User name: ').place(x = 50,y=150)
tk.Label(window,text='Password: ').place(x = 50,y=190)

var_user_name = tk.StringVar()
var_user_name.set('sample@qq.com')
entry_user_name = tk.Entry(window,textvariable = var_user_name)
entry_user_name.place(x = 160,y = 150)

var_password = tk.StringVar()
entry_password = tk.Entry(window,textvariable = var_password,show='*')
entry_password.place(x = 160,y = 190)

#  login and sign up 
def login():
    usr_name = var_user_name.get()
    usr_pwd = var_password.get()
    if usr_name == 'admin':
        if usr_pwd =='admin':
            tk.messagebox.showinfo(title='Hello',message='How are you, '+ usr_name)
        else:
            tk.messagebox.showerror(title='error',message='wrong password')
    else:
        is_sign_up = tk.messagebox.askyesno('Welcome','You have not sign up.Sign up today?')
        if is_sign_up:
            sign_up()
                
    
def sign_up():
    def sign_to_data():
        nn = new_usr_name.get()
        np = new_pwd.get()
        npf = confirm_pwd.get()

        if np!= npf:
            tk.messagebox.showwarning('Error','Password and confirm password must be the same!')
        else:
            tk.messagebox.showinfo('Welcome','You have signed up')
            #window_sign_up.destroy()

            
    window_sign_up = tk.Toplevel(window)
    window_sign_up.geometry('350x200')
    window_sign_up.title('Sign up window')
    
    new_usr_name = tk.StringVar()
    tk.Label(window_sign_up,text='user name').place(x = 10,y=10)
    tk.Entry(window_sign_up,textvariable = new_usr_name).place(x=150,y = 10)
    
    new_pwd = tk.StringVar()
    tk.Label(window_sign_up,text = 'password').place(x=10,y = 60)
    tk.Entry(window_sign_up,textvariable = new_pwd).place(x=150,y = 60)
    
    confirm_pwd = tk.StringVar()
    tk.Label(window_sign_up,text = 'confirm password').place(x=10,y = 110)
    tk.Entry(window_sign_up,textvariable = confirm_pwd).place(x=150,y = 110)
    
    signUp = tk.Button(window_sign_up,text='Sign up',command = sign_to_data)
    signUp.place(x = 150,y = 140)

btn_login = tk.Button(window,text='login',command = login)
btn_login.place(x=170,y=230)

btn_sign_up = tk.Button(window,text='sign up',command = sign_up)
btn_sign_up.place(x=270,y=230)

window.mainloop()
