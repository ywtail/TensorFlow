## 目录结构
- `cifar10_data`：数据集（代码自动生成，如果无法生成则自行下载）
- `cifar10_input.py`：下载的 TesorFlow Models 库中的，能读取本地 CIFAR-10 的二进制文件格式的内容
- `cifar10.py`：下载的 TesorFlow Models 库中的，能建立 CIFAR-10 的模型
- `cnn_book.py`：《TensorFlow实战》中的代码，准确率 71.50% （达到过73%）
- `cnn.py`：实现的 CNN，准确率 74.56%
- `cnn_lrn.py`：在上述 cnn.py 的基础上增加了 LRN，准确率 73.90%
- `cnn_l2.py`：在上述 cnn.py 的基础上对全连接层的权重进行了 L2 正则化，准确率 70.30%
- `cnn_l2_lrn.py`：在上述 cnn.py 的基础上增加了 LRN，并对全连接层的权重进行了 L2 正则化，准确率 71.90%


**详细解读见博客[TensorFlow (5): CNN对CIFAR-10进行分类](http://ywtail.github.io/2017/06/06/TensorFlow-5-CNN%E5%AF%B9CIFAR-10%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB/)**

## 参考
- 图书：TensorFlow实战 / 黄文坚，唐源著
- TensorFlow 中文社区：[卷积神经网络](http://www.tensorfly.cn/tfdoc/tutorials/deep_cnn.html)