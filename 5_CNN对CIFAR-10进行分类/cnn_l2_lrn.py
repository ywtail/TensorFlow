# coding=utf-8
# cnn_l2
from __future__ import division

import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import math
import time

data_dir = 'cifar10_data/cifar-10-batches-bin'  # 下载 CIFAR-10 的默认路径
cifar10.maybe_download_and_extract()  # 下载数据集，并解压、展开到其默认位置

batch_size = 128
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)


def weight_variable(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))  # stddev=stddev！！！
    if w1:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def bias_variable(cons, shape):
    initial = tf.constant(cons, shape=shape)  # 必须是 shape=shape
    return tf.Variable(initial)


def conv(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一层
weight1 = weight_variable([5, 5, 3, 64], 5e-2, 0.0)
bias1 = bias_variable(0.0, [64])

conv1 = tf.nn.relu(conv(image_holder, weight1) + bias1)
pool1 = max_pool_3x3(conv1)
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二层
weight2 = weight_variable([5, 5, 64, 64], 5e-2, 0.0)
bias2 = bias_variable(0.1, [64])

conv2 = tf.nn.relu(conv(norm1, weight2) + bias2)
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = max_pool_3x3(norm2)

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

# 全连接层
weight3 = weight_variable([dim, 384], 0.04, 0.004)
bias3 = bias_variable(0.1, [384])

local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 全连接层
weight4 = weight_variable([384, 192], 0.04, 0.004)
bias4 = bias_variable(0.1, [192])

local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 输出
weight5 = weight_variable([192, 10], 1 / 192.0, 0.0)
bias5 = bias_variable(0.0, [10])
logits = tf.matmul(local4, weight5) + bias5


# 损失函数
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

max_steps = 3000
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        print 'step {},loss={},({} examples/sec; {} sec/batch)'.format(step, loss_value, examples_per_sec,
                                                                       sec_per_batch)
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))  # 计算一共有多少组
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print 'precision = ', precision