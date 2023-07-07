import tensorflow as tf
from tensorflow.keras.layers import Layer


class LayerNormalization(Layer):
    """
    自定义层归一化层

    继承自Keras的Layer类，用于实现层归一化。

    参数:
        Layer (class): Keras中的Layer类

    属性:
        _epsilon (float): 防止除零的小值

    """

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建层归一化层

        参数:
            input_shape (tuple): 输入数据的形状

        """

        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        """
        调用层归一化层

        参数:
            inputs (tensor): 输入数据

        返回:
            outputs (tensor): 归一化后的输出数据

        """

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        """
        计算输出形状

        参数:
            input_shape (tuple): 输入数据的形状

        返回:
            output_shape (tuple): 输出数据的形状，与输入形状相同

        """

        return input_shape
