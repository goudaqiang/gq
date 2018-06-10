#!/usr/bin/python
#coding: utf-8
#
#功能：
#作者：gouqiang
#时间：2018/6/4
#用法：

import os
import sys
import numpy
import input_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'current path', 'mnist download path')

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir)
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b

    y_ = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    train_step = tf.train.GradientDescentOptimizer(10).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#            print 'loss%s = %s' % (i, cross_entropy.eval(feed_dict={x: batch_xs, y_: batch_ys}))

        result = sess.run(accuracy, feed_dict={x: mnist.test.images(), y_: mnist.test.labels()})
        print 'test result : %s' % result

if __name__ == '__main__':
	if len(sys.argv) == 2:
		file_dir = os.path.abspath(sys.argv[1])
	else:
		file_path = os.path.abspath(sys.argv[0])
		file_dir, _ = os.path.split(file_path)
	FLAGS.data_dir = file_dir + '/data'
	tf.app.run()

