from __future__ import print_function
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class PositionWiseFeedForward(Layer):
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        """
        位置编码前馈网络层的初始化函数。

        参数：
            model_dim: 词向量的维度。
            inner_dim: 隐藏层的维度。
            trainable: 是否可训练。
            **kwargs: 其他关键字参数。
        """
        self.bias_out = self.add_weight(
            shape=(self.model_dim,),
            initializer='uniform',
            trainable=self.trainable,
            name="bias_out")
        self.bias_inner = self.add_weight(
            shape=(self.inner_dim,),
            initializer='uniform',
            trainable=self.trainable,
            name="bias_inner")
        self.weights_out = self.add_weight(
            shape=(self.inner_dim, self.model_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name="weights_out")
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self.inner_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name="weights_inner")
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建层的权重。

        参数：
            input_shape: 输入张量的形状。
        """
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        """
        在模型调用时执行正向传播。

        参数：
            inputs: 输入张量。

        返回：
            outputs: 前馈网络输出。
        """
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bias_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bias_out
        return outputs

    def compute_output_shape(self, input_shape):
        """
        计算输出张量的形状。

        参数：
            input_shape: 输入张量的形状。

        返回：
            输出张量的形状，与输入张量相同。
        """
        return input_shape


# 示例用法
"""
query = tf.random.truncated_normal([100, 50, 150])
w = PositionWiseFeedForward(150, 2048)(query)
print(w.shape)
"""
