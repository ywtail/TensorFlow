# coding: utf-8
# Softmax Regression识别手写数字
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 创建一个新的InteractiveSession，使用这个命令会将这个session注册为默认的session，之后的运算也默认跑在这个session里
sess = tf.InteractiveSession()

# 实现softmax regression
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 使用cross_entropy作损失函数
y_ = tf.placeholder(tf.float32, [None, 10])  # y_存储实际lable
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练模型，使用梯度下降使损失函数最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

tf.global_variables_initializer().run() #设置了默认的session，可以直接这么初始化 

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次随机取100个样本进行训练
    train_step.run({x: batch_xs, y_: batch_ys}) # 设置了默认的session，可以这么写

# 模型评测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))