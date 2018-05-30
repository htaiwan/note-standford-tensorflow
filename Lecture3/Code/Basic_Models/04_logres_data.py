#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:24:12 2018

@author: htaiwan
"""
import tensorflow as tf
import time

import utils

tf.reset_default_graph()

# 定義model相關的超參數
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: 讀取資料
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: 建立datasets和iterator
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

# 將train_data & test_data 進行初始化
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# Step 3: 建立 weights 和 bias
w = tf.get_variable(name='weights', shape=(784,10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: 建立 model
logits = tf.matmul(img, w) + b

# Step 5: 定義 loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')

# Step 6: 定義 training op
# 利用 gradient descent (learning rate of 0.01) 來降低 loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step7: 計算測試集的準確度
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

# 上面step 1~7 就是建立graph，下面要開始執行graph
with tf.Session() as sess:
    
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    
    # 開始訓練model
    for i in range(n_epochs):
        # 開始從train_data讀取資料
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('epoch {0} 的平均loss {1}'.format(i, total_loss/n_batches))
    print('總花費時間: {0}'.format(time.time() - start_time))
    
    # 測試model
    sess.run(test_init)
    total_correct_preds = 0
    try:
        accuracy_batch = sess.run(accuracy)
        total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass
    
    print('準確度 {0}'.format(total_correct_preds/n_test))
writer.close()


