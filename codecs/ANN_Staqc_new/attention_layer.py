from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    """
    自定义注意力层

    继承自Keras的Layer类，用于创建一个注意力层。

    参数:
        Layer (class): Keras中的Layer类

    属性:
        kernel (tf.Variable): 注意力权重的可训练变量

    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        构建注意力层

        参数:
            input_shape (tuple): 输入数据的形状，应为长度为2的元组，每个元素的形状应为[batch_size, sequence_length, embedding_dimension]

        异常:
            ValueError: 如果输入参数不满足要求，抛出ValueError异常

        """

        # 确保输入是一个长度为2的列表，列表中每个元素的维度应为[batch_size, sequence_length, embedding_dimension]
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('An attention layer should be called on a list of 2 inputs.')

        # 确保两个输入的embedding维度相等
        if not input_shape[0][2] == input_shape[1][2]:
            raise ValueError('Embedding sizes should be of the same size')

        # 初始化权重kernel并添加到该层参数列表中
        self.kernel = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        调用注意力层

        参数:
            inputs (list): 包含两个输入的列表，每个输入形状为[batch_size, sequence_length, embedding_dimension]

        返回:
            outputs (tensor): 注意力层的输出，形状为[batch_size, sequence_length1, sequence_length2]，其中sequence_length1和sequence_length2是输入序列的长度

        """

        # 对第一个输入进行线性变换以匹配维度
        a = K.dot(inputs[0], self.kernel)
        # 对第二个输入进行维度转换以匹配第一个输入
        y_trans = K.permute_dimensions(inputs[1], (0, 2, 1))
        # 计算注意力权重
        b = K.batch_dot(a, y_trans, axes=[2, 1])
        # 对注意力权重使用双曲正切激活函数
        return K.tanh(b)

    def compute_output_shape(self, input_shape):
        """
        计算输出形状

        参数:
            input_shape (tuple): 输入数据的形状，应为长度为2的元组，每个元素的形状应为[batch_size, sequence_length, embedding_dimension]

        返回:
            output_shape (tuple): 输出数据的形状，形状为(batch_size, sequence_length1, sequence_length2)，其中sequence_length1和sequence_length2是输入序列的长度

        """

        return None, input_shape[0][1], input_shape[1][1]
