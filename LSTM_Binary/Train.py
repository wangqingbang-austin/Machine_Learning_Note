from data.utils import batch_iter
from model.LSTM_Simple import LSTM
import torch
from torch import nn, optim
import numpy as np
# Set params of model
Eopch = 200
loader, input_size = batch_iter()
# model
lstm = LSTM(input_size, 128, 2)
# loss
Loss = nn.MultiLabelSoftMarginLoss()
# optimizer
optimizer = optim.Adam(lstm.parameters(), lr=1e-3)
best_val_acc = 0
# Start the train
for epoch in range(Eopch):
    acc_train = []
    loss_train = []
    for step, (batch_x, batch_y) in enumerate(loader):
        pred = lstm(batch_x)
        # calculate the loss
        loss = Loss(pred, batch_y)
        # 3 step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss_train.append(loss.item())
        print(np.mean(loss_train))
    loss_train_mean = np.mean(loss_train)
    print('epoch:{}, loss:{}'.format(epoch, loss_train_mean))
    # 对模型进行评测
    if (epoch + 1) % 2 == 0:
        with torch.no_grad():
            acc_val = []
            for step, (x_batch, y_batch) in enumerate(loader):
                pred = lstm(x_batch)
                prob = torch.nn.functional.softmax(pred, dim=1)
                # 计算精度
                acc = np.mean((torch.argmax(prob, 1) == torch.argmax(y_batch, 1)).numpy())
                acc_val.append(acc)
            acc_mean = np.mean(acc_val)
            print('次数：', epoch, '准确度：', acc_mean)
            if acc_mean > best_val_acc and acc_mean > 0.85:
                best_val_acc = acc_mean
                torch.save(lstm.state_dict(), str(epoch) + 'lstm_params.pkl')
                print("保存模型！")

