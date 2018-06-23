#!/usr/bin/python
#coding: utf-8
#
#功能：保存模型参数
#作者：gouqiang
#时间：2018/6/21
#用法：该方式主要是直接对模型变量的恢复

import tensorflow as tf

tf.reset_default_graph()
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, './my_test_model')
    print sess.run(w1)
    print sess.run(w2)

