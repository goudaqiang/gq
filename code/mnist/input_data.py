#!/usr/bin/python
#coding: utf-8

#功能：
#作者：gouqiang
#时间：2018/6/3
#用法：直接python input_data.py是对该文件mnist数据集的下载，还有就是其他的调用

import gzip
import os
import numpy
import urllib
import sys

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

def maybe_download(filename, work_directory):
    if not os.path.exists(work_directory):  #如果不存在该目录就创建该目录
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):        #如果不存在该文件就下载
        filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print 'Succesfully downloaded ' + filename + ' %s bytes.' % statinfo.st_size
    return filepath

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')    #定义一个uint32大端模式的数据类型
    return numpy.frombuffer(bytestream.read(4), dtype=dt)   #读取4个字节，以定义的dt数据类型存放

def extract_images(filename):
    print 'Extracting' + filename
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, filename)) 
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(num_images[0] * rows[0] * cols[0])
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images[0], rows[0], cols[0], 1)
        return data

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1    #这里是使用迭代器一次性赋值，别老是想着C语言的循环了
    return labels_one_hot

def extract_labels(filename, one_hot=False):
    print 'Extracting' + filename
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items[0])
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels

####################美丽的分割线################

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
            self._num_examples = images.shape[0]
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0/255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0      #训练完成的次数记录
        self._index_in_epoch = 0        #训练的节点

    def images(self):
        return self._images

    def labels(self):
        return self._labels

    def num_examples(self):
        return self._num_examples

    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):		#用一个类方便后面的返回，后面就只返回一个类名
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    VALIDATION_SIZE = 5000		#用于验证的数据量

    local_file = maybe_download(TRAIN_IMAGES, train_dir)	#包的下载与解压处理
    train_images = extract_images(local_file)
    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)
    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)
    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)

    validation_images = train_images[:VALIDATION_SIZE]		#训练包的前一部数据分用于验证，后一部分数据用于训练
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets

if __name__ == '__main__':		#当别人调用的时候就不运行该部分，直接运行该文件测试的时候运行该部分
	file_path = os.path.abspath(sys.argv[0])
	train_dir, _ = os.path.split(file_path)
	train_dir = train_dir + '/data'
	read_data_sets(train_dir)


