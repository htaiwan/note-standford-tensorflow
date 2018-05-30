#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:08:49 2018

@author: htaiwan
"""

import tensorflow as tf 

tf.reset_default_graph()

# 範例 1: NORMAL LOADING
print('===== 範例 1 NORMAL LOADING ======')

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
# crate the op z when you assemble the graph
z = tf.add(x, y)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('graphs/normal_loading', sess.graph)
	for _ in range(10):
		sess.run(z)
	print(tf.get_default_graph().as_graph_def())
	writer.close()



# 範例 2: LAZY LOADING
print('===== 範例 2 LAZY LOADING ======')

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('graphs/lazy_loading', sess.graph)
	for _ in range(10):
        # lazy loading的陷阱，這樣的寫法會產生10個相同的add_1~10的node在graph的結構中
        # 所以在處理loss function時，千萬不能用這樣的寫法，否則grpah的size會瞬間爆炸
		sess.run(tf.add(x, y))
	print(tf.get_default_graph().as_graph_def()) 
	writer.close()

