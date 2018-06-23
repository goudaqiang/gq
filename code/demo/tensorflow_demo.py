#!/usr/bin/python
#coding: utf-8

import tensorflow as tf
import numpy as np

#a = tf.constant(1.4)
#b = tf.constant(1.5)
#c_op = tf.add(a,b)

#with tf.Session() as sess:
#    print sess.run(c_op)

###############我是漂亮的分割线#####################

#a = tf.constant(100.0)
#b = tf.Variable(0.0)
#y = tf.subtract(a,b)
#train = tf.train.GradientDescentOptimizer(0.5).minimize(y)
#init = tf.global_variables_initializer()

#with tf.Session() as sess:
#    sess.run(init)
#    for step in xrange(0,50):
#        sess.run(train)
#        print sess.run(y)

###############我是漂亮的分割线#####################

a = tf.constant(100.0)
b = tf.Variable(0.0)
y = tf.subtract(a,b)
train = tf.train.GradientDescentOptimizer(0.5).minimize(y)
init = tf.global_variables_initializer()

#tf.summary.scalar('a', a)
b_summary = tf.summary.scalar('b', b)
#merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/tensorboard_logs', tf.get_default_graph())

sess = tf.Session()
sess.run(init)
for step in xrange(0,50):
    sess.run(train)
    result = sess.run(b_summary)
    writer.add_summary(result, step)
    print sess.run(y)
sess.close()


