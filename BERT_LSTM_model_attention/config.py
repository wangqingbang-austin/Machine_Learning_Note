START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
# glove的位置，绝对路径,提前准备好了转化为word2vec使用
glove_path = 'C:\\Users\\wangq\\Desktop\\EI\\实验\\LSTM_model_word2vec\\data\\baike_26g_news_13g_novel_229g.bin'

# 是否为中文文本，默认为英文
isChinese = True
# 选取字典的大小,中文建议5000，英文建议12000
if isChinese:
    vocab_size = 10000
else:
    vocab_size = 12000

batch_size = 50
# 词表的位置
vocab_dir = 'data/vocab.txt'