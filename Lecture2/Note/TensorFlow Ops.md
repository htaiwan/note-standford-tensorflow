#  TensorFlow Ops

## Agenda

- **Basic operations**
	-  [Your first TensorFlow program](#1)
	-  [Visualize it with TensorBoard](#2)
	-  [Explicitly name them](#3)
- **Tensor types**
   -  [Constants](#4)
   		-  [Tensors filled with a specific value](#5)
   		-  [Constants as sequences](#6)
   		-  [Randomly Generated Constants](#7)
   -  [Operations](#8)
   		-  [Arithmetic Ops](#9)
   		-  [Wizard of Div](#10)
   -  [TensorFlow Data Types](#11)
   		- [TF vs NP Data Types](#12)
   		- [Use TF DType when possible](#13)
   	- [What’s wrong with constants?](#14)
   	- [Variables](#15)
   		-  [tf.Variable class](#16)
   		-  [You have to initialize your variables](#17)
   		-  [Eval() a variable](#18)
   		-  [tf.Variable.assign()](#19)
   		-  [assign_add() and assign_sub()](#20)
   		-  [Each session maintains its own copy of variables](#21)
   		-  [Control Dependencies](22)

- **Importing data**
   	-  [Placeholder](#23)
   		-  [Why placeholders?](#24)
   		-  [Placeholders are valid ops](#25)
   		-  [What if want to feed multiple data points in?](#26)
   		-  [Feeding values to TF ops](#27)

- **Lazy loading**
	-  [What’s lazy loading?](#28)
   		-  [Lazy loading Example](#20)
   		-  [tf.get_default_graph().as_graph_def()](#30)
   	-  [The trap of lazy loading](#31)

## Note

<h2 id="1">Your first TensorFlow program</h2>

![1](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/1.png)

- 編譯時會出現警告 !! [參考](http://blog.xuite.net/abliou/linux/486757306-使用tensorflow的sse3+sse4.1+sse4.2指令集)
- 如何避免警告。

<h2 id="2">Visualize it with TensorBoard</h2>

![2](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/2.png)

- 在完成grpah定義後，準備開始執行session之前，建立summary writer。
- 路徑可以隨意指定，並非一定要按照上面的例子，將log file存於指定的路徑下。
- 執行tensorboard，將log file讀到tensorboard。

```
tensorboard --logdir="./graphs" --port 6006
```

- 打開browser，直接到 [​http://localhost:6006/​]( http://localhost:6006/​ 

<h2 id="3">Explicitly name them</h2>

![3](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/3.png)

- 賦予更精確的命名，以便在tensorboad進行找尋識別。

<h2 id="4">Constants</h2>

![4](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/4.png)

- tf常數的宣告方式。
- 跟Numpy一樣，具有broadCasting的功能。

<h2 id="5">Tensors filled with a specific value</h2>

![5](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/5.png)

- 建立一個tensor後，直接給tensor中每個值都賦予一個預設值。

<h2 id="6">Constants as sequences</h2>

![6](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/6.png)

- 建立一個具有連續值的tensor。

<h2 id="7">Randomly Generated Constants</h2>

![7](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/7.png)

- 建立一個具有隨機值的tensor。

<h2 id="8">Operations</h2>

![8](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/8.png)

- tf的基本運算元的分類。

<h2 id="9">Arithmetic Ops</h2>

![9](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/9.png)

- tf的數學運算元。

<h2 id="10">Wizard of Div</h2>

![10](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/10.png)

- 特別注意的是除法運算元，又細分了很多不同的除法。
- [document](https://www.tensorflow.org/api_docs/python/tf/divide)

<h2 id="11">TensorFlow Data Types</h2>

![11](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/11.png)

<h2 id="12">TF vs NP Data Types</h2>

![12](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/12.png)

<h2 id="13">Use TF DType when possible</h2>

![13](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/13.png)

<h2 id="14">What’s wrong with constants?</h2>

![14](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/14.png)

<h2 id="15">Variables</h2>

![15](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/15.png)

<h2 id="16">tf.Variable class</h2>

![16](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/16.png)

<h2 id="17">You have to initialize your variables</h2>

![17](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/17.png)

<h2 id="18">Eval() a variable</h2>

![18](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/18.png)

<h2 id="19">tf.Variable.assign()</h2>

![19](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/19.png)

![20](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/20.png)

![21](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/21.png)

<h2 id="20">assign_add() and assign_sub()</h2>

![22](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/22.png)

<h2 id="21">Each session maintains its own copy of variables</h2>

![23](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/23.png)

<h2 id="22">Control Dependencies</h2>

![24](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/24.png)

<h2 id="23">Placeholder </h2>

![25](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/25.png)

<h2 id="24">Why placeholders?</h2>

![26](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/26.png)

<h2 id="25">Placeholders are valid ops</h2>

![27](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/27.png)

![28](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/28.png)

![29](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/29.png)

<h2 id="26">What if want to feed multiple data points in?</h2>

![30](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/30.png)

<h2 id="27">Feeding values to TF ops</h2>

![31](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/31.png)

![32](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/32.png)

<h2 id="28">What’s lazy loading?</h2>

![33](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/33.png)


<h2 id="29">Lazy loading Example</h2>

![34](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/34.png)

<h2 id="30">tf.get_default_graph().as_graph_def()</h2>

![35](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/35.png)

<h2 id="31">The trap of lazy loading</h2>

![36](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/36.png)

![37](https://github.com/htaiwan/note-standford-tensorflow/blob/master/Lecture2/Assets/37.png)




