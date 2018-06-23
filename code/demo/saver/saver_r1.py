#!/usr/bin/python
#coding: utf-8
#
#功能：读取模型参数
#作者：gouqiang
#时间：2018/6/23
#用法：该方式主要是通过读取图的方式来对变量进行恢复

import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./my_test_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")

    print sess.run(w1)
    print sess.run(w2)






