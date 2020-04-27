from data.utils import batch_iter
from model.LSTM_Simple import LSTM
import torch
from torch import nn, optim
import numpy as np

loader, input_size = batch_iter()
# model
lstm = LSTM(input_size, 128, 2)

lstm.load_state_dict(torch.load('model/131lstm_params.pkl'))

with torch.no_grad():
    count = 0
    acc_val = []
    for step, (x_batch, y_batch) in enumerate(loader):
        pred = lstm(x_batch)
        prob = torch.nn.functional.softmax(pred, dim=1)
        result = np.array(torch.argmax(prob, 1) == torch.argmax(y_batch, 1))
        result = [i for i in result if i]
        count += len(result)
        # 计算精度
        acc = np.mean((torch.argmax(prob, 1) == torch.argmax(y_batch, 1)).numpy())
        acc_val.append(acc)
    acc_mean = np.mean(acc_val)
    print('准确度：', acc_mean, count)
