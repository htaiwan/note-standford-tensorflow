#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:28:03 2018

@author: htaiwan
"""

from tensorflow.contrib.tensorboard.plugins import projector
import os
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

class SkipGramModel:    
    def __init__(self, dataset, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
    
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False, dtype=tf.int32)
        self.skip_step = SKIP_STEP

    def _import_data(self):
            dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
            self.iterator = dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()
        # step 1: 輸入資料 (create dataset and generate sample from them)
        # 利用name_scope將目的相似的node分組
 #       with tf.name_scope('data'):          
            #self.iterator = self.dataset.make_initializable_iterator()
            #self.center_words, self.target_words = self.iterator.get_next()
        
    def _create_embedding(self):
        # step 2: 定義權重 (Define the weight)
        # step 3: 定義模型結構 Inference(compute the forward path of the graph)
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable('embed_matrix',
                                                shape=[self.vocab_size, self.embed_size],
                                                initializer=tf.random_uniform_initializer())
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embedding')
           
    def _create_loss(self):
        # step 4: 定義損失函數 (Define the loss function)
        with tf.name_scope('loss'):
            # construct variables for NCE loss
            nce_weight = tf.get_variable('nce_weight',
                                         shape=[self.vocab_size, self.embed_size],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0 / (self.embed_size ** 0.5)))
            nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
            
            # define loss function to be NCE loss function
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                                biases=nce_bias, 
                                                labels=self.target_words, 
                                                inputs=self.embed, 
                                                num_sampled=self.num_sampled, 
                                                num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        # step 5: 定義最佳化函數
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
     
    def _create_summaries(self):
        # 利用tensorboard來觀察更細節的統計數字的呈現
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            # 因為會有多個summaries, 所以最好merget成單一個op，會比較方便管理
            self.summary_op = tf.summary.merge_all()
    
    def build_graph(self):
        # 替model建立graph
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
          
    def train(self, num_train_steps):
        # 利用tf.train.Saver定期將訓練中的參數進行紀錄
        # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias
        saver = tf.train.Saver()
        initial_step = 0
        utils.safe_mkdir('checkpoints')
        
        with tf.Session() as sess:
            # 宣告變數共享
            tf.get_variable_scope().reuse_variables() 
            
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
                        
            # 如果已經有checkpoint的紀錄，那就從記錄開始，否則就從頭開始
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            # 利用這個來計算當前SKIP_STEP的平均loss
            total_loss = 0.0 
            # 根據不同實驗的learning rate記錄不同的summary檔案
            writer = tf.summary.FileWriter('graphs/word2vec/lr' + str(self.lr), sess.graph)
            initial_step = self.global_step.eval()
            
            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('在step {}: 平均損失 {:5.1f}'.format(index, total_loss / self.skip_step))
                        total_loss = 0.0
                        # 記錄當前的訓練參數
                        saver.save(sess, 'checkpoints/skip-gram', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()
        

    def visualize(self, visual_fld, num_visualize):
        """ run "'tensorboard --logdir='visualization'" to see the embeddings """
        
        # 建立最常用的word清單來進行視覺話
        word2vec_utils.most_common_words(visual_fld, num_visualize)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)
            
            # you have to store embeddings in a new variable
            embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embedding_var.initializer)
            
            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_fld)
            
            # add embedding to the config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            
            # link this tensor to its metadata file, in this case the first NUM_VISUALIZE words of vocab
            embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

            # saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)
                        
def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, 
                                        BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)

def main():
    dataset = tf.data.Dataset.from_generator(gen, 
                                (tf.int32, tf.int32), 
                                (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    model = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    model.visualize(VISUAL_FLD, NUM_VISUALIZE)

if __name__ == '__main__':
    main()