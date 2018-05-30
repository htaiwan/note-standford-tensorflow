#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 09:48:40 2018

@author: htaiwan
"""

import tensorflow as tf

import utils
import word2vec_utils

# Model hyperparameters
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128            # dimension of the word embedding vectors
SKIP_WINDOW = 1             # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000        # number of tokens to visualize

tf.reset_default_graph()

# Phase 1: 建立grpah
def word2vec():

# step 1: 輸入資料 (create dataset and generate sample from them)
    def gen():
        yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, 
                                        BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)
    dataset = tf.data.Dataset.from_generator(gen, 
                                             (tf.int32, tf.int32), 
                                             (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    
    iterator = dataset.make_initializable_iterator()
    center_words, target_words = iterator.get_next()
    
    # 確認格式
    with tf.Session() as sess:
        sess.run(iterator.initializer)     
        # 數字表示在字典中的對應的index位置
        # print('center_words 格式 \n {0}'.format(center_words.eval()))
        # print('target_words 格式 \n {0}'.format(target_words.eval()))
    

# step 2: 定義權重 (Define the weight)
    # words(VOCAB_SIZE) * feature vector(EMBED_SIZE)
    embed_matrix = tf.get_variable('embed_matrix',
                                   shape=[VOCAB_SIZE, EMBED_SIZE],
                                   initializer=tf.random_normal_initializer())
    
    # 確認格式
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 隨機初始化 weight matrix
        # print(embed_matrix.eval())
        
# step 3: 定義模型結構 Inference(compute the forward path of the graph)
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

# step 4: 定義損失函數 (Define the loss function)
    nce_weight = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE],
                                 initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))
    nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
    
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                         biases=nce_bias,
                                         labels=target_words,
                                         inputs=embed,
                                         num_sampled=NUM_SAMPLED, 
                                         num_classes=VOCAB_SIZE), name='loss')
    
     # 確認格式
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print(nce_weight.eval())
        # print(nce_bias.eval())


# step 5: 定義最佳化函數
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Phase 2: 執行計算
   
    with tf.Session() as sess:
        # step 1: 第一次要先初始化model中所有的變數
        sess.run(iterator.initializer)
                
        # step 2: 初始化iterator並讀取training data
        sess.run(tf.global_variables_initializer())
        
        # step 3: 根據training data開始訓練model
        writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)
        total_loss = 0.0
        for index in range(NUM_TRAIN_STEPS):
            try:
                # step 4: 計算損失值
                # step 5: 調整參數
                loss_batch, _ = sess.run([loss, optimizer])
                total_loss += loss_batch
                if (index + 1) % SKIP_STEP == 0: # 每5000次打印一次目前的平均loss
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                    total_loss = 0.0
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
        writer.close()

def main():
    word2vec()

if __name__ == '__main__':
    main()
    

