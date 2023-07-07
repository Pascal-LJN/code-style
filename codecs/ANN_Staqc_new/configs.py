# 跑模型运行的配置
def get_config(train):
    """
    获取配置参数

    参数:
        train (str): 训练类型

    返回:
        dict: 配置参数字典
    """

    # 配置参数字典
    conf = {
        'workdir': '../train_data/new/origin_model/' + train + '/',
        'buckets': [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)],
        'data_params': {
            # 训练数据路径
            "train_path": '../data/new_data_hnn/' + train + '/hnn_' + train + '_train_f.pkl',
            # 验证数据路径
            'valid_path': '../data/new_data_hnn/' + train + '/hnn_' + train + '_dev_f.pkl',
            # 测试数据路径
            'test_path': '../data/new_data_hnn/' + train + '/hnn_' + train + '_test_f.pkl',

            'code_pretrain_emb_path': '../data/new_data_hnn/' + train + '/' + train + '_word_vocab_final.pkl',
            'text_pretrain_emb_path': '../data/new_data_hnn/' + train + '/' + train + '_word_vocab_final.pkl'
        },
        'training_params': {
            'batch_size': 100,
            'nb_epoch': 150,

            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'reload': 0,  # 如果reload=0，则从头开始训练
            'dropout1': 0,
            'dropout2': 0,
            'dropout3': 0,
            'dropout4': 0,
            'dropout5': 0,
            'regularizer': 0,

        },

        'model_params': {
            'model_name': 'CodeMF',
        }
    }
    return conf


# 打标签运行的配置
def get_config_u2l(train):
    """
    获取配置参数（用于U2L模型）

    参数:
        train (str): 训练类型

    返回:
        dict: 配置参数字典
    """

    # 配置参数字典
    conf = {
        'workdir': '../train_data/new/fianl/code_sa/' + train + '/',
        'buckets': [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)],
        'data_params': {
            # 训练数据路径
            "train_path": '../data/new_data_hnn/' + train + '/hnn_' + train + '_train_f.pkl',
            # 验证数据路径
            'valid_path': '../data/new_data_hnn/' + train + '/hnn_' + train + '_dev_f.pkl',
            # 测试数据路径
            'test_path': '../data/new_data_hnn/' + train + '/hnn_' + train + '_test_f.pkl',

            # 词向量路径
            'code_pretrain_emb_path': '../data_processing/hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl',
            'text_pretrain_emb_path': '../data_processing/hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
        },
        'training_params': {
            'batch_size': 100,
            'nb_epoch': 150,

            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'reload': 0,  # 如果reload=0，则从头开始训练
            'dropout1': 0,
            'dropout2': 0,
            'dropout3': 0,
            'dropout4': 0,
            'dropout5': 0,
            'regularizer': 0,

        },

        'model_params': {
            'model_name': 'CodeMF',
        }
    }
    return conf
