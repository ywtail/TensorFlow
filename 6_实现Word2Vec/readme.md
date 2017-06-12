## 目录结构
- `text8.zip`：数据集（代码自动下载，如果无法下载则手动下载：http://mattmahoney.net/dc/text8.zip ）
- `word2vec.py`：实现的 CNN，准确率 74.56%
- `cnn_lrn.py`：在上述 cnn.py 的基础上增加了 LRN，准确率 73.90%
- `cnn_l2.py`：在上述 cnn.py 的基础上对全连接层的权重进行了 L2 正则化，准确率 70.30%
- `cnn_l2_lrn.py`：在上述 cnn.py 的基础上增加了 LRN，并对全连接层的权重进行了 L2 正则化，准确率 71.90%


**详细解读见博客[TensorFlow (5): CNN对CIFAR-10进行分类](http://ywtail.github.io/2017/06/06/TensorFlow-5-CNN%E5%AF%B9CIFAR-10%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB/)**

## 参考
- 图书：TensorFlow实战 / 黄文坚，唐源著
- TensorFlow 中文社区：[卷积神经网络](http://www.tensorfly.cn/tfdoc/tutorials/deep_cnn.html)