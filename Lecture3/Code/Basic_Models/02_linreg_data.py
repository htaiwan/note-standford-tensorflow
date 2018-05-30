#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:18:05 2018

@author: htaiwan
"""

import time

import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'data/birth_life_2010.txt'

# fix Variable bias already exists, 
# disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
tf.reset_default_graph()

# Step 1: 從.txt檔案中讀取數據
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# **Step 2**: (重要的地方) 建立 Dataset 和 iterator
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

# Step 3: 建立 weight 和 bias, 並設定初始值 0.0
# Make sure to use tf.get_variable
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: 建立 model 來預測 Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = w * X + b

# Step 5: 利用 square error 來當作 loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: 利用 gradient descent 搭配 learning rate 0.001 來降低 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

with tf.Session() as sess:
    # Step 7: 初始化所需要的變數 w 和 b
    sess.run(tf.global_variables_initializer()) 
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)


    # Step 8: 訓練 model, 100 epochs
    for i in range(100):
        # (重要的地方) 在每次loop的時候，都初始化iterator 
        sess.run(iterator.initializer)
        
        total_loss = 0
        # (重要的地方)
        try:
            while True:
                _, l = sess.run([optimizer, loss]) 
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass
            
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    # 關閉writer
    writer.close()
    
    # Step 9: 輸出最終的w和b的值
    w_out, b_out = sess.run([w, b])
    print('w: %f, b: %f' %(w_out, b_out))

print('花費: %f 秒' %(time.time() - start))

# uncomment the following lines to see the plot 
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
