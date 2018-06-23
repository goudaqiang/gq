#!/usr/bin/python
#coding: utf-8
#
#功能：读取模型参数
#作者：gouqiang
#时间：2018/6/21
#用法：

import tensorflow as tf

tf.reset_default_graph()
w1 = tf.Variable(tf.zeros([2]), name='w1')
w2 = tf.Variable(tf.zeros([5]), name='w2')
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print sess.run(w1)
    print sess.run(w2)
