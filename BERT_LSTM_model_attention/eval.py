import torch.autograd as autograd
import torch.optim as optim
import torch
import torch.nn as nn
from BLSTM_CRF import *
import config
from data_utils import  batch_iter, build_vocab
import os
import numpy as np
from tqdm import tqdm


base_dir = 'data'
train_sen_dir = os.path.join(base_dir, 'train_sen_data.txt')
train_BIO_dir = os.path.join(base_dir, 'train_BIO_label_data.txt')
test_sen_dir = os.path.join(base_dir, 'test_sen_data.txt')
test_BIO_dir = os.path.join(base_dir, 'test_BIO_label_data.txt')

# 加载
training_data = batch_iter(train_sen_dir, train_BIO_dir)
test_data = batch_iter(test_sen_dir, test_BIO_dir)
# 加载词典
word_to_ix = build_vocab()
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, config.EMBEDDING_DIM, config.HIDDEN_DIM)
model.load_state_dict(torch.load('14lstm_params.pkl'))

with torch.no_grad():
    total = []
    with open('eval1.txt', 'w', encoding='utf8') as f:
        for sentence, tags in tqdm(training_data):
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            score, tag_seq = model(sentence_in)
            tol = []
            for index, i in enumerate(tag_seq):
                if i == 1:
                    tol.append(sentence[index])
                elif i == 0:
                    tol.append(sentence[index])
            f.write(' '.join(tol) + '\n')
            total.append(tol)
