import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class SelfAttention(Layer):
    def __init__(self, r, da, name, **kwargs):
        """
        自注意力机制层的初始化函数。

        参数：
            r: 注意力头数。
            da: 注意力隐藏层维度。
            name: 层的名称。
            **kwargs: 其他关键字参数。
        """
        self.r = r
        self.da = da
        self.scope = name
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建层的权重。

        参数：
            input_shape: 输入张量的形状。
        """
        # 初始化权重矩阵 Ws1 和 Ws2
        # Ws1: (input_shape[2], self.da)
        self.Ws1 = self.add_weight(name='Ws1_' + self.scope,
                                   shape=(input_shape[2], self.da),
                                   initializer='glorot_uniform',
                                   trainable=True)

        # Ws2: (self.da, self.r)
        self.Ws2 = self.add_weight(name='Ws2_' + self.scope,
                                   shape=(self.da, self.r),
                                   initializer='glorot_uniform',
                                   trainable=True)

    def call(self, inputs, **kwargs):
        """
        在模型调用时执行正向传播。

        参数：
            inputs: 输入张量。
            **kwargs: 其他关键字参数。

        返回：
            B: 自注意力输出。
            P: 自注意力权重的正则化项。
        """
        # inputs: (batch_size, seq_len, input_dim)

        # 计算 A1
        # A1: (?, seq_len, self.da)
        A1 = K.dot(inputs, self.Ws1)
        A1 = tf.transpose(A1, perm=[0, 2, 1])  # 转置成 (batch_size, self.da, seq_len)
        A1 = tf.tanh(A1)  # 经过激活函数

        # 计算注意力权重 A_T
        # A_T: (?, seq_len, self.r)
        A_T = K.softmax(K.dot(A1, self.Ws2))
        A = tf.transpose(A_T, perm=[0, 2, 1])  # 转置成 (batch_size, self.r, seq_len)

        # 计算 B
        # B: (batch_size, self.r, input_dim)
        B = tf.matmul(A, inputs)

        # 计算 P
        tile_eye = tf.tile(tf.eye(self.r), [tf.shape(inputs)[0], 1])
        tile_eye = tf.reshape(tile_eye, [-1, self.r, self.r])  # tile_eye: (batch_size, self.r, self.r)
        AA_T = tf.matmul(A, A_T) - tile_eye  # AA_T: (batch_size, self.r, self.r)
        P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))  # P: (batch_size,)

        return [B, P]

    def compute_output_shape(self, input_shape):
        """
        计算输出张量的形状。

        参数：
            input_shape: 输入张量的形状。

        返回：
            以列表形式返回输出张量的形状。
        """
        return [(input_shape[0], self.da, self.r), (input_shape[0],)]
