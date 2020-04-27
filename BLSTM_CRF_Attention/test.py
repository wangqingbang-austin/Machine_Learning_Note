import torch
import numpy as np
import tqdm

total = []
with open('eval.txt', 'r', encoding='utf8') as f:
    for i in f:
        total.append(i.strip().split(' '))

labels = []
count = 0
acc = 0
with open('data/test_label_data_name.txt', 'r', encoding='utf8') as f1:
    for i in f1:
        labels.append(i.strip('\n').split('\t'))

    for index, sen in enumerate(labels):
        flag = True
        for label in sen:
            if label in total[index]:
                count += 1
            else:
                flag = False
        if flag:
            acc += 1
print(len(labels))
print(count)
print(acc)
print(count / (2 * len(labels)))




# a = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])
# b = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
# # c = np.mean((a == b).numpy())
# # print(c)
#
# a = np.argwhere(a.numpy() != 2)
# b = np.argwhere(b.numpy() != 2)
# print(a)
# print(b)
# c = np.mean((a == b).numpy(), axis=0)
# print(c)
# for i in tqdm(a):
#     print(i)
# print(np.mean(a))
