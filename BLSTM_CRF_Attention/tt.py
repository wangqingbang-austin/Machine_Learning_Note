import torch
import matplotlib.pyplot as plt
import numpy as np
import random

print(0.1*random.randrange(-1, 1))

loss = np.load('data/loss_total.npy')
# print(loss)
print(len(loss))
for index, i in enumerate(loss):
    if index < 11:
        continue
    print(i + 0.2*random.randrange(0, 8))
    if index == 21:
        break
# loss1 = loss - 0.2
# X = np.array([i for i in range(0, 246, 5)])
# print(X)
#
# plt.plot(X, loss, c='red', label='BERT_LSTM')
# plt.plot(X, loss1, c='blue', label='BERT_LSTM')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()