# coding: utf-8
# Softmax Regression识别手写数字
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 打印数据集基本信息
print '=' * 30
print 'train:', mnist.train.images.shape, mnist.train.labels.shape
print 'test:', mnist.test.images.shape, mnist.test.labels.shape
print 'validation:', mnist.validation.images.shape, mnist.validation.labels.shape
print '=' * 30

# 实现softmax regression模型: y=softmax(xW+b)
x = tf.placeholder(tf.float32, [None, 784])  # x 使用占位符，在后续输入时填充
W = tf.Variable(tf.zeros([784, 10]))  # W 和 b 参数使用Variable，在迭代过程中不断更新
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 即 y = softmax(xW+b)

# 用cross_entropy作损失函数
y_ = tf.placeholder(tf.float32, [None, 10])  # y_表示 label的实际值
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 实现交叉熵函数

# 训练模型
# 使用梯度下降最小化损失函数cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()  # 对所有的参数进行初始化
sess = tf.Session()  # 在一个Session里运行模型
sess.run(init)  # 执行初始化

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次随机取100个样本进行训练
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 判断预测值与真实值是否相等，返回的correct_prediction是一个布尔值的列表，例如 [True, False, True, True]。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 将correct_prediction输出的bool值转换为float，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 输出模型在测试及上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
