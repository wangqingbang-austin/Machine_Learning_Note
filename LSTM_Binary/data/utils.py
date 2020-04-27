import torch
import torch.utils.data as Data
import re


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass


def build_vocab(sentence_data):
    """
    :param sentence_data:
    :return: two dict about word and idx
    """
    word2idx = {0: '<PAD>'}
    idx2word = {'<PAD>': 0}
    count = 1
    for line in sentence_data:
        for word in line:
            word = re.sub('[^\w\u4e00-\u9fff]+', '', word)
            if is_number(word) or word == '':
                continue
            if word in word2idx:
                continue
            word2idx[word] = count
            idx2word[count] = word
            count += 1
    return word2idx, idx2word


def get_data():
    """
    read the data from corpus file
    :return:
    """
    x_data, y_data = [], []
    with open('data/corpus.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            if float(line[0]) != 0 and float(line[0]) != 1:
                continue
            label = int(line[0])
            sentence = line[0:]
            x_data.append(sentence[:2000])
            y_data.append(label)
    print("The length of dataset is:", len(x_data))
    return x_data, y_data


def get_tensor_data(x_data, y_data):
    """
    :return: the format of the tensor data
    """
    word2idx, _ = build_vocab(x_data)
    x_data_idx = []
    for sentence in x_data:
        idx = [word2idx[word] for word in sentence if word in word2idx]
        x_data_idx.append(idx)
    x_tensor_data = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in x_data_idx], batch_first=True)
    t = torch.tensor(y_data).view(len(y_data), 1)
    y_tensor_data = torch.zeros(len(t), 2).scatter_(1, t, 1)
    return x_tensor_data, y_tensor_data, len(word2idx)


def batch_iter():
    """

    :param batch_size: returns a specified amount of data
    :param x_data:
    :param y_data:
    :return: a iter
    """
    x_data, y_data = get_data()
    x_tensor_data, y_tensor_data, input_size = get_tensor_data(x_data, y_data)
    dataset = Data.TensorDataset(x_tensor_data, y_tensor_data)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=1000,
        shuffle=True,
    )
    return loader, input_size
