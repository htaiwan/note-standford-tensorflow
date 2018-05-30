#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:20:48 2018

@author: htaiwan
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

tf.reset_default_graph() 

# 定義model相關的超參數
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# Step 1: 讀取資料
# 利用TF Learn's 的內建function幫我們處理 MNIST 數據的輸入
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(batch_size)

# Step2: 建立feature(X)和labels(Y)的placehoder
# MNIST data中每張圖片的大小 28*28 = 784
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
# 每張圖片要對應的 1~10 classes,對應的數字 0~9
Y = tf.placeholder(tf.int32, [batch_size, 10], name='label')

# Step3: 建立weights和bias
# w 是隨機初始化 mean of 0, stddev of 0.01
# w 的shape 是要mapping X (784)-> Y (10)
w = tf.get_variable(name='weight', shape=(784, 10), initializer=tf.random_normal_initializer())
# b 初始化 = 0
# b 的shape 跟Y的shape相同
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())


# Step4: 建立model
logits = tf.matmul(X, w) + b

# Step5: 定義loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
# 計算整個batch的平均loss
loss = tf.reduce_mean(entropy)
# loss = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(logits) * tf.log(Y), reduction_indices=[1]))

# Step6: 定義training op
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# Step7: 計算測試集的準確度
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg_placeholder', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())	
    n_batches = int(mnist.train.num_examples/batch_size)
    
    # 訓練model n_epochs次
    for i in range(n_epochs):
        total_loss = 0
            
        for j in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], {X: X_batch, Y:Y_batch}) 
            total_loss += loss_batch
        print('epoch {0} 平均損失值: {1}'.format(i, total_loss/n_batches))
    print('總花費時間: {0} 秒'.format(time.time() - start_time))
    
    # 測試 the model
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run(accuracy, {X: X_batch, Y:Y_batch})
        total_correct_preds += accuracy_batch	
        
    print('準確度 {0}'.format(total_correct_preds/mnist.test.num_examples))

writer.close()
        