#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 21:00:40 2018

@author: htaiwan
"""

import tensorflow as tf

# fix: ValueError: Variable scalar already exists, disallowed. 
# Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
tf.reset_default_graph()

# 範例 1: 快速建立log file writer
print('===== 範例 1 ======')

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
# 建立writer
writer = tf.summary.FileWriter('./graphs/simple', tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(x))
# 記得不用時要將writer關閉
writer.close()

# 範例 2: 好多好多的除法啊
print('===== 範例 2 ======')

a = tf.constant([2,2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
with tf.Session() as sess:
    print(sess.run(tf.div(b,a)))
    print(sess.run(tf.divide(b, a)))
    print(sess.run(tf.floordiv(b, a)))
    # print(sess.run(tf.realdiv(b, a)))
    print(sess.run(tf.truncatediv(b, a)))
    print(sess.run(tf.floor_div(b, a)))

# 範例 3: 乘法
print('===== 範例 3 ======')

a = tf.constant([10, 20], name='a')
b = tf.constant([2, 3], name='b')

with tf.Session() as sess:
    print(sess.run(tf.multiply(a, b)))
    print(sess.run(tf.tensordot(a, b, 1)))
    

# 範例 4: Python native type
print('===== 範例 4 ======')

t_0 = 19
x = tf.zeros_like(t_0)
y = tf.ones_like(t_0)

t_1 = ['apple', 'peach', 'banana']
x = tf.zeros_like(t_1) 					# ==> ['' '' '']
# y = tf.ones_like(t_1) 				# ==> TypeError: Expected string, got 1 of type 'int' instead.

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]] 
x = tf.zeros_like(t_2) 					# ==> 3x3 tensor, all elements are False
y = tf.ones_like(t_2) 					# ==> 3x3 tensor, all elements are True

print(tf.int32.as_numpy_dtype())
