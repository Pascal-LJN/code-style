from __future__ import print_function
import os
import tensorflow as tf
import sys
import numpy as np
import random
import pickle
import argparse
import logging
from sklearn.metrics import *
from configs import *
import warnings

warnings.filterwarnings("ignore")

'''
tf.compat.v1.set_random_seed(1)  # 图级种子，使所有操作会话生成的随机序列在会话中可重复，请设置图级种子：
random.seed(1)  # 让每次生成的随机数一致
np.random.seed(1)  #
'''
set_session = tf.compat.v1.keras.backend.set_session
# 配置GPU内存分配
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.55  # half of the memory
set_session(tf.compat.v1.Session(config=config))

# 设置随机种子
random.seed(42)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


class StandoneCode:
    """
    自定义StandoneCode类

    用于加载数据、填充数据、保存/加载模型和其他处理操作。

    参数:
        conf (dict): 配置参数字典，默认为None

    属性:
        conf (dict): 配置参数字典
        _buckets (list): 数据分桶的大小列表
        _buckets_text_max (tuple): 文本输入数据分桶的最大长度
        _buckets_code_max (tuple): 代码输入数据分桶的最大长度
        path (str): 工作目录路径
        train_params (dict): 训练参数字典
        data_params (dict): 数据参数字典
        model_params (dict): 模型参数字典
        _eval_sets (None): 评估集合（未使用）

    """

    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self._buckets = conf.get('buckets', [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)])
        self._buckets_text_max = (max([i for i, _, _, _ in self._buckets]), max([j for _, j, _, _ in self._buckets]))
        self._buckets_code_max = (max([i for _, _, i, _ in self._buckets]), max([j for _, _, _, j in self._buckets]))
        self.path = self.conf.get('workdir', './data/')
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())
        self._eval_sets = None

    def load_pickle(self, filename):
        """
        加载pickle文件

        参数:
            filename (str): pickle文件路径

        返回:
            word_dict (dict): 加载的pickle文件内容

        """

        with open(filename, 'rb') as f:
            word_dict = pickle.load(f)
        return word_dict

    ##### Padding #####

    def pad(self, data, len=None):
        """
        填充数据序列

        使用pad_sequences函数对数据序列进行填充。

        参数:
            data (list): 数据序列列表
            len (int): 填充后的最大长度，默认为None

        返回:
            padded_data (ndarray): 填充后的数据序列

        """

        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Model Loading / saving #####

    def save_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        """
        保存模型和参数

        将模型保存到指定的文件路径。

        参数:
            model: 要保存的模型对象
            epoch (int): 当前的迭代轮数
            d12, d3, d4, d5, r: 模型参数

        """

        if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.path + 'models/' + self.model_params['model_name'] + '/')
        model.save("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.path,
                                                                                                  self.model_params[
                                                                                                      'model_name'],
                                                                                                  d12, d3, d4, d5, r,
                                                                                                  epoch),
                   overwrite=True)

    def load_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        """
        加载指定轮次的模型权重

        参数:
            model: 要加载权重的模型对象
            epoch (int): 要加载的轮次
            d12, d3, d4, d5, r: 权重文件参数

        异常:
            FileNotFoundError: 如果找不到指定轮次的权重文件

        """
        # 构建权重文件路径
        weights_file_path = "{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
            self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch)

        # 检查权重文件是否存在
        assert os.path.exists(weights_file_path), "未找到第 {:d} 轮的权重".format(epoch)

        # 加载权重到模型中
        model.load(weights_file_path)

    def del_pre_model(self, prepoch, d12, d3, d4, d5, r):
        """
        删除倒数第二个轮次的模型权重文件

        参数:
            prepoch (list): 包含多个轮次的列表
            d12, d3, d4, d5, r: 权重文件参数

        """
        if len(prepoch) >= 2:
            # 获取倒数第二个轮次
            epoch = prepoch[-2]

            # 构建权重文件路径
            weights_file_path = "{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
                self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch)

            # 检查权重文件是否存在，如果存在则删除
            if os.path.exists(weights_file_path):
                os.remove(weights_file_path)

    def process_instance(self, instance, target, maxlen):
        """
        处理单个实例，并将结果添加到目标列表中

        参数:
            instance: 要处理的实例数据
            target (list): 保存处理结果的目标列表
            maxlen (int): 最大长度

        """
        w = self.pad(instance, maxlen)
        target.append(w)

    def process_matrix(self, inputs, trans1_length, maxlen):
        """
        处理输入矩阵，返回二维列表

        参数:
            inputs: 要处理的输入矩阵
            trans1_length (int): 切割长度
            maxlen (int): 最大长度

        返回:
            processed_inputs (list): 处理后的二维列表

        """
        inputs_trans1 = np.split(inputs, trans1_length, axis=1)
        processed_inputs = []
        for item in inputs_trans1:
            item_trans2 = np.squeeze(item, axis=1).tolist()
            processed_inputs.append(item_trans2)
        return processed_inputs

    def get_data(self, path):
        """
        从指定路径加载数据并进行处理

        参数:
            path (str): 数据文件的路径

        返回:
            text_S1 (list): 处理后的文本 S1 列表
            text_S2 (list): 处理后的文本 S2 列表
            code (list): 处理后的代码列表
            queries (list): 原始查询列表
            labels (list): 原始标签列表
            ids (list): 原始 ID 列表

        """
        # 加载数据
        data = self.load_pickle(path)

        # 初始化各个列表
        text_S1 = []
        text_S2 = []
        code = []
        queries = []
        labels = []
        ids = []

        # 定义处理参数
        text_block_length = 2
        text_word_length = 100
        query_word_length = 25
        code_token_length = 350

        # 处理文本块数据
        text_blocks = self.process_matrix(np.array([samples_term[1] for samples_term in data]),
                                          text_block_length, text_word_length)
        text_S1 = text_blocks[0]
        text_S2 = text_blocks[1]

        # 处理代码块数据
        code_blocks = self.process_matrix(np.array([samples_term[2] for samples_term in data]),
                                          text_block_length - 1, code_token_length)
        code = code_blocks[0]

        # 处理查询、标签和ID数据
        queries = [samples_term[3] for samples_term in data]
        labels = [samples_term[5] for samples_term in data]
        ids = [samples_term[0] for samples_term in data]

        return text_S1, text_S2, code, queries, labels, ids

    ##### Training #####
    def train(self, model):
        """
        训练模型

        参数:
            model: 要训练的模型实例

        """

        # 获取训练参数
        nb_epoch = self.train_params.get('nb_epoch', 20)

        for i in range(nb_epoch):
            print('Epoch %d :: \n' % i, end='')

            # 加载训练数据
            text_S1, text_S2, code, queries, labels, _ = self.get_data(self.data_params['train_path'])

            # 在当前 epoch 上训练模型一次
            hist = model.fit(
                [np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)], np.array(labels),
                shuffle=True, epochs=1, batch_size=self.train_params.get('batch_size', 100))

            # 计算并记录训练集的指标
            train_acc, train_f1, train_recall, train_precison, train_loss = self.valid(model,
                                                                                       self.data_params['train_path'])
            print("train data: %d loss=%.3f, acc=%.3f, precison=%.3f, recall=%.3f, f1=%.3f" % (
                i, train_loss, train_acc, train_precison, train_recall, train_f1))

            # 计算并记录验证集的指标
            dev_acc, dev_f1, dev_recall, dev_precision, dev_loss = self.valid(model, self.data_params['valid_path'])
            print("dev data: %d loss=%.3f, acc=%.3f, precison=%.3f, recall=%.3f, f1=%.3f" % (
                i, dev_loss, dev_acc, dev_precision, dev_recall, dev_f1))

            # 计算并记录测试集的指标
            test_acc, test_f1, test_recall, test_precision, test_loss = self.valid(model, self.data_params['test_path'])
            print("test data: %d loss=%.3f, acc=%.3f, precison=%.3f, recall=%.3f, f1=%.3f" % (
                i, test_loss, test_acc, test_precision, test_recall, test_f1))

        sys.stdout.flush()

    def valid(self, model, path):
        """
        quick validation in a code pool.
        param:
            poolsize - size of the code pool, if -1, load the whole test set
        """

        text_S1, text_S2, code, queries, labels, _ = self.get_data(path)

        labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                                  batch_size=100)
        labelpred = np.argmax(labelpred, axis=1)
        loss = log_loss(labels, labelpred)
        acc = accuracy_score(labels, labelpred)
        f1 = f1_score(labels, labelpred)
        recall = recall_score(labels, labelpred)
        precision = precision_score(labels, labelpred)
        return acc, f1, recall, precision, loss

        ##### Evaluation in the develop set #####

    def eval(self, model, path):
        """
        evaluate in a evaluation date.
        param:
            poolsize - size of the code pool, if -1, load the whole test set
        """

        text_S1, text_S2, code, queries, labels, ids = self.get_data(path)

        labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],
                                  batch_size=100)
        labelpred = np.argmax(labelpred, axis=1)

        loss = log_loss(labels, labelpred)
        acc = accuracy_score(labels, labelpred)
        f1 = f1_score(labels, labelpred)
        recall = recall_score(labels, labelpred)
        precision = precision_score(labels, labelpred)
        print("相应的测试性能: precision=%.3f, recall=%.3f, f1=%.3f, accuracy=%.3f" % (
            precision, recall, f1, acc))
        return acc, f1, recall, precision, loss


# https://wenku.baidu.com/view/5101dd03cfbff121dd36a32d7375a417866fc19f.html
'''
name or flags - 选项字符串的名字或者列表，例如 foo 或者 -f, --foo
choices:参数可允许的值的⼀个容器
default - 不指定参数时的默认值。
help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表⽰不显⽰该参数的帮助信息.
action - 命令⾏遇到参数时的动作，默认值是 store
'''


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Model")  # 创建对象
    parser.add_argument("--train", choices=["python", "sql"], default="python", help="train dataset set")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluat models in a test set ")  # 添加参数
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    conf = get_config(args.train)
    train_path = conf['data_params']['train_path']
    dev_path = conf['data_params']['valid_path']
    test_path = conf['data_params']['test_path']

    ##### Define model ######
    logger.info('Build Model')
    # 这里是不是相当于创建了CARLCS_CNN类，并用conf里面的参数去初始化它
    model = eval(conf['model_params']['model_name'])(
        conf)  # initialize the model,  model== <models.CARLCS_CNN object at 0x7f1d9c2e2cc0>

    StandoneCode = StandoneCode(conf)
    drop1 = drop2 = drop3 = drop4 = drop5 = 0.8

    r = 0.0002

    conf['training_params']['regularizer'] = 8
    model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4,
                        dropout5=drop5,
                        Regularizer=round(r, 5), num=8,
                        seed=42)
    conf['training_params']['dropout1'] = drop1
    conf['training_params']['dropout2'] = drop2
    conf['training_params']['dropout3'] = drop3
    conf['training_params']['dropout4'] = drop4
    conf['training_params']['dropout5'] = drop5
    conf['training_params']['regularizer'] = round(r, 5) + 1
    model.build()

    if args.mode == 'train':
        StandoneCode.train(model)
    elif args.mode == 'eval':

        # 加载模型python
        # StandoneCode.load_model_epoch(model, 121, 0.5, 0.5, 0.5, 0.5, 0.0006)
        ## 加载模型sql
        StandoneCode.load_model_epoch(model, 83, 0.25, 0.25, 0.25, 0.25, 0.0006)
        # 测试集评估
        StandoneCode.eval(model, test_path)
    # for d in np.arange(0.49,0.51,0.01):
