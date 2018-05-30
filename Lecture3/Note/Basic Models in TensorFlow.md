# Basic Models in TensorFlow

## Agenda

- **Review**
	-  [Computation graph](#1)
	-  [TensorBoard](#2)
	-  [tf.constant and tf.Variable](#3)
	-  [tf.placeholder and feed_dict](#4)
	-  [Avoid lazy loading](#5)
- **Linear Regression in TensorFlow**
   -  [Background](#6)
   -  [Interactive Coding](#7)
   -  [Huber loss](#8)
   -  [TF Control Flow](#9)  
- **tf.data**
   	-  [Placeholder](#10)
   	-  [tf.data](#11)
-  **Optimizers**
	-  [Optimizer](#12)
	-  [Trainable variables](#13)
	-  [List of optimizers in TF](#14)
- **Logistic Regression in TensorFlow**
	-  Background
		-  MNIST
		-  Model
	-  Interactive Coding


## Note

<h2 id="1">Computation graph</h2>

![1](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture3/Assets/1.png)

> - 運算圖 (將執行過程中的所有運算逐步切割)。
> - 兩個重要步驟:
> 	- 建立運算圖。
>  	- 執行運算圖。

<h2 id="1">TensorBoard</h2>

![2](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture3/Assets/2.png)