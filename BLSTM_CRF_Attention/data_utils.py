import os
from collections import Counter
import torch
import numpy as np
import torch.utils.data as Data
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import config
from tqdm import tqdm
# , max_length=config.max_length
def batch_iter(train_name, label_name):
    """
    生成批次处理数据
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    # 读需要处理的文本
    contents, labels = [], []
    # 读需要处理的文本
    with open(train_name, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            contents.append(line.strip('\n').split(' '))
        f.close()
    # 读需要处理文本的标签
    with open(label_name, 'r', encoding='UTF-8') as f1:
        for line in f1.readlines():
            labels.append(line.strip('\n').split(' '))
        f.close()

    training_data = []
    for index, sen in enumerate(contents):
        tup_data = (sen, labels[index])
        training_data.append(tup_data)

    return training_data

def build_vocab():
    '''

    :return:
    '''
    if not os.path.exists('pretrained/vocab.npy'):
        base_dir = 'data'
        train_sen_dir = os.path.join(base_dir, 'train_sen_data.txt')
        train_BIO_dir = os.path.join(base_dir, 'train_BIO_label_data.txt')
        test_sen_dir = os.path.join(base_dir, 'test_sen_data.txt')
        test_BIO_dir = os.path.join(base_dir, 'test_BIO_label_data.txt')
        # 加载
        training_data = batch_iter(train_sen_dir, train_BIO_dir)
        test_data = batch_iter(test_sen_dir, test_BIO_dir)
        # 加载词典
        word_to_ix = {}
        for sentence, tags in tqdm(training_data):
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        for sentence, tags in tqdm(test_data):
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        np.save('pretrained/vocab.npy', word_to_ix)
    else:
        # Load
        word_to_ix = np.load('pretrained/vocab.npy', allow_pickle=True).item()
    print("加载词典完毕！")
    return word_to_ix

def read_vocab(vocab_dir=config.vocab_dir):
    '''
    读取已经创建好的词汇表,并构建word2id, id2word
    :return: word2id, id2word
    '''
    # 读取词汇表

    with open(vocab_dir, 'r', encoding='UTF-8') as f:
        words = [word.strip() for word in f.readlines()]
    # 构建word2id => dict(zip(['one', 'two', 'three'], [1, 2, 3])) 映射函数方式来构造字典
    word2id = dict(zip(words, range(len(words))))
    id2word = dict(zip(range(len(words)), words))
    return id2word, word2id


def get_embedding_weight(glove_path=config.glove_path, vocab_size=config.vocab_size,
                         embedding_dim=config.EMBEDDING_DIM):
    '''
    pytorch引入词向量
    :param glove_path:  这里已经提前转好为word2vec向量
    :param vocab_size:
    :param embedding_dim:
    :return:
    '''
    # 这里用的是word2vec词向量
    if not os.path.exists('pretrained/word2vec_weight.pkl'):
        wvmodel = KeyedVectors.load_word2vec_format(glove_path, binary=True)
        weight = torch.zeros(vocab_size + 1, embedding_dim)
        id2word, word2id = read_vocab()
        for i in range(len(wvmodel.index2word)):
            try:
                index = word2id[wvmodel.index2word[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy(wvmodel.get_vector(
                id2word[word2id[wvmodel.index2word[i]]]))
        torch.save(weight, 'pretrained/word2vec_weight.pkl')
    else:
        weight = torch.load('pretrained/word2vec_weight.pkl')
    return weight

if __name__ == '__main__':
    base_dir = 'data'
    train_sen_dir = os.path.join(base_dir, 'train_sen_data.txt')
    train_BIO_dir = os.path.join(base_dir, 'train_BIO_label_data.txt')
    # 加载数据
    _, word_to_ix = read_vocab()
    training_data = batch_iter(train_sen_dir, train_BIO_dir)
    print()