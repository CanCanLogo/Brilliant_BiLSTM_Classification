import pandas as pd
import random
import csv
from tqdm import tqdm
import numpy as np
import operator
import os
import pickle as pkl

# ## 创建英文词典
def build_vocab(sentences, verbose=True):
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def clean_special_chars(text, punct, mapping, contraction_mapping):
    for p in contraction_mapping:
        text = text.replace(p, contraction_mapping[p])
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    return text

# 加载词向量
def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    return embeddings_index

# ## 检查预训练embeddings和vocab的覆盖情况
def check_coverage(vocab, embeddings_index):
    known_words = {}  # 两者都有的单词
    unknown_words = {}  # embeddings不能覆盖的单词
    nb_known_words = 0  # 对应的数量
    nb_unknown_words = 0
    #     for word in vocab.keys():
    for word in tqdm(vocab):
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))  # 覆盖单词的百分比
    print('Found embeddings for  {:.2%} of all text'.format(
        nb_known_words / (nb_known_words + nb_unknown_words)))  # 覆盖文本的百分比，与上一个指标的区别的原因在于单词在文本中是重复出现的。
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    print("unknown words : ", unknown_words[:30])
    return unknown_words


def build_vocab(file_path, max_size, min_freq):
    df = pd.read_csv(file_path, encoding='utf-8', sep=';')
    # 转化为小写
    sentences = df['content'].apply(lambda x: x.lower())
    # 去除特殊字符
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    sentences = sentences.apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    # 提取数组
    sentences = sentences.progress_apply(lambda x: x.split()).values
    vocab_dic = {}
    for sentence in tqdm(sentences, disable=False):
        for word in sentence:
            try:
                vocab_dic[word] += 1
            except KeyError:
                vocab_dic[word] = 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"词典======== {vocab}")

def get_embed(vocab_path, embed_path, dim):
    vocab = pkl.load(open(vocab_path, 'rb'))
    embed_glove = load_embed(embed_path)
    ebed = []
    for v in vocab:
        if v not in embed_glove.keys():
            ebed.append(np.asarray([0 for i in range(0, dim)], dtype='float32'))
        else:
            ebed.append(embed_glove[v])
    return np.asarray(ebed, dtype='float32')

class Config():
    def __init__(self):
        self.vocab_path = 'vocab.pkl'
        self.train_path = 'train.csv'
        self.dev_path = 'dev.csv'
        self.test_path = 'test.csv'
        self.pad_size = 16


# punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
#           '+', '\\', '•', '~', '@', '£',
#           '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
#           '½', 'à', '…',
#           '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
#           '▓', '—', '‹', '─',
#           '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
#           'Ã', '⋅', '‘', '∞',
#           '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
#           '≤', '‡', '√', ]
#
# punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
#                  "—": "-", "–": "-", "’": "'", "_": "-", "`": "'",
#                  '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha',
#                  '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '',
#                  '³': '3', 'π': 'pi', }

df = pd.read_csv("labelled_newscatcher_dataset.csv", encoding='utf-8', sep=';')

# # ## 进度条初始化
# tqdm.pandas()
# # ## 加载数据集
# df = pd.read_csv("labelled_newscatcher_dataset.csv", encoding='utf-8', sep=';')
# # 注意要转为小写，并清除词语
# sentences = df['title'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
# sentences = sentences.apply(lambda x: x.lower()).progress_apply(lambda x: x.split()).values
# # ## 创建词典
# vocab = build_vocab(sentences)
#
# # 加载预训练词向量
# glove = 'glove.6B.50d.txt'
# embed_glove = load_embed(glove)
#
# # 检测覆盖率
# oov_glove = check_coverage(vocab, embed_glove)


# 获得vocab
# tqdm.pandas()
MAX_VOCAB_SIZE = 7000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
# build_dataset(Config())

vocab_path = 'vocab.pkl'
embed_path = 'glove.6B.300d.txt'
dim = 300
np.savez('glove.6B.300d.npz', embeddings=get_embed(vocab_path, embed_path, dim))
