#!/usr/bin/python
#coding: utf-8
#
#功能：
#作者：gouqiang
#时间：2018/6/4
#用法：

import os
import sys
import input_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'enter path', 'mnist download path')

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        init = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
        w_conv1 = tf.Variable(init)
        init = tf.constant(0.1, shape=[32])
        b_conv1 = tf.Variable(init)
        conv_1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = tf.nn.relu(conv_1 + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv2'):
        init = tf.truncated_normal([5, 5, 32, 64], stddev=0.1)
        w_conv2 = tf.Variable(init)
        init = tf.constant(0.1, shape=[64])
        b_conv2 = tf.Variable(init)
        conv_2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(conv_2 + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('fc1'):
        init = tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1)
        w_fc1 = tf.Variable(init)
        init = tf.constant(0.1, shape=[1024])
        b_fc1 = tf.Variable(init)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        fc_1 = tf.matmul(h_pool2_flat, w_fc1)
        h_fc1 = tf.nn.relu(fc_1 + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        init = tf.truncated_normal([1024, 10], stddev=0.1)
        w_fc2 = tf.Variable(init)
        init = tf.constant(0.1, shape=[10])
        b_fc2 = tf.Variable(init)
        fc_2 = tf.matmul(h_fc1_drop, w_fc2)
        h_fc1 = tf.nn.relu(fc_2 + b_fc2) 

    return h_fc1, keep_prob



def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int64, [None])

    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)    #强制类型转换
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
                print 'step %d, training accuracy %g' % (i, train_accuracy)
            train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

        print 'test  accuracy %g' % accuracy.eval(feed_dict={x:mnist.test.images(), y_:mnist.test.labels(), keep_prob:1.0})


if __name__ == '__main__':
    if len(sys.argv) == 2:
	file_dir = os.path.abspath(sys.argv[1])
    else:
    	file_path = os.path.abspath(sys.argv[0])
    	file_dir, _ = os.path.split(file_path)
    FLAGS.data_dir = file_dir + '/data'
    tf.app.run()

