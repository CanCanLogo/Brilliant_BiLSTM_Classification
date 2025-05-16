import torch
import numpy as np
class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding):
        self.train_path = dataset + 'train.csv'  # 训练集
        self.dev_path = dataset + 'valid.csv'  # 验证集
        self.test_path = dataset + 'test.csv'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + 'class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + 'vocab.pkl'  # 词表
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量

        self.model_name = 'BiLSTM'
        self.save_path = dataset + '/saved_model/'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.pad_size = 16  # 短填长切
        self.max_token_per_sent = 32
        self.dropout = 0.1  # dropout 当num_layers=1,dropout是无用的
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # batch大小
        self.learning_rate = 0.0001  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.weight_decay = 5e-4 # weight decay