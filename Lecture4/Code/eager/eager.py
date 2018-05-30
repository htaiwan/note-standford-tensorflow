#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:06:20 2018

@author: htaiwan
"""

import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

import utils

DATA_FILE = 'data/birth_life_2010.txt'

# 要使用eager,必須在tf program的最開始就要宣告了
# 看起來若跑過後，就要刪掉這行，不然就要restart kernel cmd + . ?
# https://blog.csdn.net/omodao1/article/details/80277079
tfe.enable_eager_execution()

# 讀取資料到dataset
data, n_samples = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

# 建立變數
# 利用tfe.Variable 而不是 tf.Variable
w = tfe.Variable(0.0)
b = tfe.Variable(0.0)

# Define the linear predictor.
def prediction(x):
  return x * w + b

# 定義loss function
def squared_loss(y, y_predicted):
    return (y - y_predicted) ** 2

def huber_loss(y, y_predicted, m=1.0):
    t = y - y_predicted
    # 因為啟用了eager execution所以在這裡可以直接使用python control flow
    # 不然就一定要使用tf的control flow ex. tf.condition
    return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

def train(loss_fn):
    print('訓練: loss function名稱: ' + loss_fn.__name__)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    # 計算每次的預測結果誤差
    def loss_for_example(x, y):
        return loss_fn(y, prediction(x))
    
    # 計算loss function的微分
    # `grad_fn(x_i, y_i)` returns 
    # (1) the value of `loss_for_example` evaluated at `x_i`, `y_i`
    # (2) the gradients of any variables used in calculating it.
    # Returns a function which differentiates f with respect to variables
    grad_fn = tfe.implicit_value_and_gradients(loss_for_example)
    
    start = time.time()
    for epoch in range(100):
        total_loss = 0.0
        # 利用tfe.Iterator
        for x_i, y_i in tfe.Iterator(dataset):
            loss, gradients = grad_fn(x_i, y_i)
            # Take an optimization step and update variables.
            optimizer.apply_gradients(gradients)
            total_loss += loss
        # 每10次列印目前的平均loss
        if epoch % 10 == 0:
            print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))
    print('總花費時間: %f 秒' %(time.time() - start))
    print('Eager execution exhibits significant overhead per operation. '
        'As you increase your batch size, the impact of the overhead will '
        'become less noticeable. Eager execution is under active development: '
        'expect performance to increase substantially in the near future!')
   

train(huber_loss)
plt.plot(data[:,0], data[:,1], 'bo')
# The `.numpy()` method of a tensor retrieves the NumPy array backing it.
# In future versions of eager, you won't need to call `.numpy()` and will
# instead be able to, in most cases, pass Tensors wherever NumPy arrays are
# expected.
plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r',
         label="huber regression")
plt.legend()
plt.show()