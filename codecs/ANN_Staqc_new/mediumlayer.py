from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import backend as K


class MediumLayer(Layer):
    def __init__(self, **kwargs):
        super(MediumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MediumLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        定义层的前向传播逻辑。

        Args:
            inputs: 输入张量
            **kwargs: 其他参数

        Returns:
            sentence_token_level_outputs: 处理后的输出张量
        """
        sentence_token_level_outputs = tf.stack(inputs, axis=0)
        sentence_token_level_outputs = K.permute_dimensions(sentence_token_level_outputs, (1, 0, 2))
        return sentence_token_level_outputs

    def compute_output_shape(self, input_shape):
        """
        计算输出张量的形状。

        Args:
            input_shape: 输入张量的形状

        Returns:
            output_shape: 输出张量的形状
        """
        return (input_shape[0][0], len(input_shape), input_shape[0][1])
