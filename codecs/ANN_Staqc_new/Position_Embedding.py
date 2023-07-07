import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        """
        位置编码层的初始化函数。

        参数：
            size: 位置编码的长度，必须为偶数。
            mode: 编码模式，可以是 'sum' 或 'concat'。
            **kwargs: 其他关键字参数。
        """
        self.size = size
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        """
        在模型调用时执行正向传播。

        参数：
            x: 输入张量。

        返回：
            position_ij: 位置编码后的张量。
        """
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])

        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]

        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)

        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1
        position_i = K.expand_dims(position_i, 2)

        position_ij = K.dot(position_i, position_j)
        position_ij_2i = K.sin(position_ij)[..., tf.newaxis]
        position_ij_2i_1 = K.cos(position_ij)[..., tf.newaxis]
        position_ij = K.concatenate([position_ij_2i, position_ij_2i_1])
        position_ij = K.reshape(position_ij, (batch_size, seq_len, self.size))

        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


# 示例用法
"""
query = tf.random.truncated_normal([100, 50, 150])
w = Position_Embedding(150, 'concat')(query)
print(w.shape)
"""
