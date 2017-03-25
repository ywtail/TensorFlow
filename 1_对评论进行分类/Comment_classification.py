# coding=utf-8
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import random
import numpy as np


# 获得分词结果
def create_tokens(filename):
    tokens = []
    lines = open(filename, 'r').readlines()
    for line in lines:
        tokens += word_tokenize(line)
    return tokens


# 整理词汇
# 使用lemmatize进行词性还原
# 使用FreqDist进行词频统计，统计完成相当于去重完成
# 沿用参考博客的词汇筛选方法：选词频>20且<2000的词放入词汇表lex中
def create_lexicon(lex):
    wnl = WordNetLemmatizer()
    lex = [wnl.lemmatize(w) for w in lex]
    lex_freq = nltk.FreqDist(lex)
    print('词性还原及去重后词汇数：', len(lex_freq))
    lex = [w for w in lex_freq if lex_freq[w] > 20 and lex_freq[w] < 2000]
    print('词频在20到2000之间的词汇数：', len(lex))
    return lex


# 评论向量化
# 每条评论向量的维数是len(lex),初始化为全0，若评论中的词在lex中存在，则词汇对应位置为1
# lex是词汇表，clf是评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
def create_dataset(filename, lex, clf):
    lines = open(filename, 'r').readlines()
    dataset = []
    for line in lines:
        features = [0 for i in range(len(lex))]
        words = word_tokenize(line)
        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(w) for w in words]
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        dataset.append([features, clf])
    return dataset


# 构造神经网络
# 此处构建的是具有两层hidden layer的前馈神经网络
def neural_network(data, n_input_layer, n_layer_1, n_layer_2, n_output_layer):
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# 使用数据训练神经网络
# epochs=15，训练15次
def train_neural_network(X, Y, train_dataset, test_dataset, batch_size, predict):
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    epochs = 15
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        random.shuffle(train_dataset)
        train_x = train_dataset[:, 0]
        train_y = train_dataset[:, 1]

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(train_x) - batch_size)[::batch_size]:
                batch_x = train_x[i:i + batch_size]
                batch_y = train_y[i:i + batch_size]

                _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
                epoch_loss += c
            print(epoch, 'epoch_loss :', epoch_loss)

        test_x = test_dataset[:, 0]
        test_y = test_dataset[:, 1]
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率：', accuracy.eval({X: list(test_x), Y: list(test_y)}))


def main():
    pos_file = 'pos.txt'
    neg_file = 'neg.txt'

    lex = []  
    lex += create_tokens(pos_file) #正面评论分词
    lex += create_tokens(neg_file)
    print('分词后词汇数：', len(lex))

    lex = create_lexicon(lex) # 词汇整理

    dataset = []  # 保存评论向量化结果
    dataset += create_dataset(pos_file, lex, [1, 0])  # 正面评论
    dataset += create_dataset(neg_file, lex, [0, 1])  # 负面评论
    print('总评论数：', len(dataset))

    random.shuffle(dataset)
    dataset = np.array(dataset)

    test_size = int(len(dataset) * 0.3)  # 取30%的数据作为测试数据集
    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    n_input_layer = len(lex)
    n_layer_1 = 1000
    n_layer_2 = 1000
    n_output_layer = 2
    batch_size = 50  # 每次取50条评论进行训练

    X = tf.placeholder('float', [None, len(train_dataset[0][0])])
    Y = tf.placeholder('float')
    predict = neural_network(X, n_input_layer, n_layer_1, n_layer_2, n_output_layer)
    train_neural_network(X, Y, train_dataset, test_dataset, batch_size, predict)


if __name__ == '__main__':
    main()

"""
运行结果：

分词后词汇数： 230193
词性还原及去重后词汇数： 18643
词频在20到2000之间的词汇数： 1065
总评论数： 10662
0 epoch_loss : 48243.6735458
1 epoch_loss : 11543.6250602
2 epoch_loss : 4191.00249064
3 epoch_loss : 2044.00308001
4 epoch_loss : 1885.96657889
5 epoch_loss : 1615.12769464
6 epoch_loss : 246.230073355
7 epoch_loss : 115.469640963
8 epoch_loss : 79.0952082175
9 epoch_loss : 119.461824983
10 epoch_loss : 86.1617076425
11 epoch_loss : 60.006764943
12 epoch_loss : 130.805916614
13 epoch_loss : 137.929141973
14 epoch_loss : 98.6247218253
准确率： 0.612258
"""
