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

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

# *******准备数据
# 文件
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
#***********


# ******** 准备模型
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, config.EMBEDDING_DIM, config.HIDDEN_DIM)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# ********

# ******* 开始训练
# Make sure prepare_sequence from earlier in the LSTM section is loaded
print("模型开始训练！")
best_acc = 0.3
best_loss = 10
loss_total = []
for epoch in range(150):
    loss_epoch = []
    for sentence, tags in tqdm(training_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        sentence_in = sentence_in.to(device)
        targets = targets.to(device)
        # Step 3. Run our forward pass.

        loss = model.neg_log_likelihood(sentence_in, targets)
        loss_epoch.append(loss.item())

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()

        loss.backward()
        optimizer.step()
    print(epoch, np.mean(loss_epoch))
    loss_total.append(np.mean(loss_epoch))
    if (epoch+1) % 5 == 0:
        np.save("data/loss_total.npy", loss_total)
        with torch.no_grad():
            acc_total = []
            for sentence, tags in tqdm(test_data):
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
                sentence_in = sentence_in.to(device)
                targets = targets.to(device)
                score, tag_seq = model(sentence_in)
                tag_seq = torch.tensor(tag_seq)
                acc_sen = np.mean((tag_seq == targets.to('cpu')).numpy())
                acc_total.append(acc_sen)
            acc = np.mean(acc_total)
            print('次数：', epoch+1, '准确度：', acc, 'loss:',  np.mean(loss_epoch))
            if np.mean(loss_epoch) < best_loss:
                best_loss = np.mean(loss_epoch)
                torch.save(model.state_dict(), str(epoch) + 'lstm_params.pkl')
                print("保存最新模型！")
