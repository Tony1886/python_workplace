# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 9:24
# @Author  : Tan Zhijie
# @Email   : tanzj@siom.ac.cn
# @File    : test01.py
# @Software: PyCharm

# If you want to execute your code on a quantum chip:
My_token =  'f8513242b744d000bf38e991fab133505f6b8535ae48af9215f08a36c798a85446b833b7ea9045af5aea6c5a3601c5b8db43b052df88fcbce85ae833b42e4942'
from qiskit import IBMQ
IBMQ.save_account(My_token)
