#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 08:22:39 2018

@author: htaiwan
"""

import tensorflow as tf

tf.reset_default_graph()

# 範例 1: feed_dict with placeholder
print('===== 範例 1 feed_dict with placeholder ======')

# a 是placehodler, 有3個elements, 其type是tf.float32
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b

writer = tf.summary.FileWriter('graphs/placeholders', tf.get_default_graph())
with tf.Session() as sess:
    # feed_dict
    print(sess.run(c, {a: [1, 2, 3]}))
writer.close()


# 範例 2: feed_dict with variables
print('===== 範例 2 feed_dict with variables ======')
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(b, feed_dict={a: 15}))
