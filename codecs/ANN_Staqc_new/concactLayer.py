import tensorflow as tf
from tensorflow.keras.layers import Layer


class ConcatLayer(Layer):
    """
    自定义拼接层

    继承自Keras的Layer类，用于将输入数据在第二维度进行拼接。

    参数:
        Layer (class): Keras中的Layer类

    """

    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建拼接层

        参数:
            input_shape (tuple): 输入数据的形状

        """

        super(ConcatLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        调用拼接层

        参数:
            inputs (tensor): 输入数据

        返回:
            outputs (tensor): 拼接后的输出数据

        """

        # 将输入数据按照第二维度进行分割
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        # 将分割后的数据拼接起来形成一个大的张量
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        # 将第二维度移除，将张量变为二维
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        return block_level_code_output

    def compute_output_shape(self, input_shape):
        """
        计算输出形状

        参数:
            input_shape (tuple): 输入数据的形状

        返回:
            output_shape (tuple): 输出数据的形状

        """

        return input_shape[0], input_shape[1] * input_shape[2]
