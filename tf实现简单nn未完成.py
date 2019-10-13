# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:46:34 2019

@author: Bird L
"""

###tensorflow在模拟数据集实现简单神经网络
import tensorflow as tf


###通过numpy生成数据集
from numpy.random import RandomState

###定义batch大小
batch_size = 8

###定义神经网络参数
w1 = tf.Variable(tf.random.normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev = 1, seed = 1))
###stddev表示随机数的标准为1，还可以通过参数mean来指定随机数的均值，默认为0

###使shape的一个维度为none可以方便使用不同大小的batch
x = tf.compat.v1.placeholder(tf.float32, shape = (None, 2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 3), name = 'y-input')

###定义神经网络向前传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

###定义损失函数和反向传播算法
y = tf.sifmoid(y)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

###生成模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

###创建会话来运行tensorflow
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    ##初始化变量
    sess.run(init_op)
    
    print(sess.run(w1))
    print(sess.run(w2))






