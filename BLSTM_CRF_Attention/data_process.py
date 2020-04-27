from tqdm import tqdm

def set_BIO_label():
    '''

    :return:
    '''
    sen_data = []
    with open('data/test_sen_data.txt', 'r', encoding='UTF-8') as f1:
        for line in f1:
            sen_data.append(line.strip('\n'))
    label_data = []
    with open('data/test_label_data_name.txt', 'r', encoding='UTF-8') as f2:
        for line in f2:
            label_data.append(line.strip('\n'))
    sen_BIO_label = []
    for index, line in enumerate(sen_data):
        BIO_label = []
        sen = line.split(' ')
        for word in sen:
            if word in label_data[index]:
                BIO_label.append('B')
            else:
                BIO_label.append('O')
        sen_BIO_label.append(BIO_label)
    with open('data/test_BIO_label_data.txt', 'w', encoding='UTF-8') as f3:
        for line in sen_BIO_label:
            f3.write(' '.join(line) + '\n')
    print("标注成功")

if __name__ == '__main__':
    set_BIO_label()