#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:39:23 2018

@author: htaiwan
"""

import tensorflow as tf

# fix: ValueError: Variable scalar already exists, disallowed. 
# Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
tf.reset_default_graph()

# 範例 1: 列印 graph's definition
print('===== 範例 1 列印 graph definition ======')

# constant & variable 不同的地方
# constant 是存於 graph's definition
my_const = tf.constant([1.0, 2.0], name='my_const')
print(tf.get_default_graph().as_graph_def())

# 範例 2: 建立變數
print('===== 範例 2 建立變數 ======')

# 舊方式
# 變數V是大寫, 常數c是小寫
s = tf.Variable(2, name='scalar') 
m = tf.Variable([[0, 1], [2, 3]], name='matrix') 
W = tf.Variable(tf.zeros([784,10]), name='big_matrix')
V = tf.Variable(tf.truncated_normal([784, 10]), name='normal_matrix')

# 新方式（easy variable sharing ??）
s = tf.get_variable('scalar', initializer=tf.constant(2)) 
m = tf.get_variable('matrix', initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable('big_matrix', shape=(784, 10), initializer=tf.zeros_initializer())
V = tf.get_variable('normal_matrix', shape=(784, 10), initializer=tf.truncated_normal_initializer())

# 範例 3: 初始化變數
print('===== 範例 3 初始化變數 ======')

with tf.Session() as sess:
    # 一次初始化所有變數
    sess.run(tf.global_variables_initializer())
    # 初始化部分變數
    # sess.run(tf.variables_initializer[s, m])
    # 只初始化某個特定變數
    # sess.run(W.initializer)

# 範例 4: 檢查變數值
print('===== 範例 4 檢查變數值 ======')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 如果沒初始化就檢查，會出現error: FailedPreconditionError
    print(sess.run(V))
    print(V.eval())

# 範例 5: 給變數指定值
print('===== 範例 5 給變數指定值 ======')

W = tf.Variable(10)
# 這樣的寫法並不會讓W變成100
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W))

W = tf.Variable(10)
# 這樣的寫法才是正確的
assign_op = W.assign(100)
with tf.Session() as sess:
    # 不用特別初始化，因為assign的動作已經自動幫忙進行初始化了
    sess.run(assign_op)
    print(W.eval())      

# tf.Variabe.assign_add(sub)並不會自動幫忙初始化，要自己來
W = tf.Variable(10)

with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W.assign_add(10)))
    print(sess.run(W.assign_sub(2)))
    

# 範例 6: 每個session都擁有各自獨立的變數
print('===== 範例 6 每個session都擁有各自獨立的變數 ======')

W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10)))        	# >> 20
print(sess2.run(W.assign_sub(2)))        	# >> 8
print(sess1.run(W.assign_add(100)))        	# >> 120
print(sess2.run(W.assign_sub(50)))        	# >> -42
sess1.close()
sess2.close()

# 範例 7: 建立具有相依性的變數
print('===== 範例 7 建立具有相依性的變數 ======')

C = tf.Variable(tf.truncated_normal([700, 10]))
# 這樣的寫法會出現 FailedPreconditionError:
# U = tf.Variable(W * 2) 
# 正確寫法，要先初始化相依的變數
U = tf.Variable(C.initialized_value() * 2)
  