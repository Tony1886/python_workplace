# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:59:06 2018

@author: Tony
"""

#import tensorflow as tf
import sympy  as sy

#L = tf.Variable(tf.float32)
#f = tf.Variable(tf.float32)
#f = tf.placeholder(tf.float32)
#
#d2 = f/(1+f)
#
#sess = tf.Session()
#
#sess.run(tf.global_variables_initializer())
#
#print(sess.run(d2,feed_dict={f:3.}))

L = sy.Symbol('L')
l = 0.2e-2
f = 20e-2
d1 = 1
h = 200e-6
s = 50e-2
d2 = f*d1/(d1-f)

a_x = d2
a_y = (L-h/2)/d1*d2+L

b_x = d2
b_y = (L+h/2)/d1*d2+L

c_x = 0
c_y = L+l/2

d_x = 0
d_y = L-l/2

kac = (a_y-c_y)/(a_x-c_x)
kbd = (b_y-d_y)/(b_x-d_x)

ymin = kac*(d2+s)+c_y
ymax = kbd*(d2+s)+d_y

results = sy.solve(ymin-0.207e-2,L)

print(results)


results = [0e-2]
print('illumination area: ',(ymax-ymin).evalf(subs={L :results[0]})*1e3,'mm')
print('kac:',kac.evalf(subs={L :results[0]}))
print('kbd:',kbd.evalf(subs={L :results[0]}))
print('ymin:',ymin.evalf(subs={L :results[0]}))
print('a_y:',a_y.evalf(subs={L :results[0]}))
# =============================================================================
# results = sy.solve(ymin-l1-0.5e-2,L)
# print(results)
# l1 = (ymax-ymin).evalf(subs={L :results[0]})
# print(l1)
# =============================================================================

