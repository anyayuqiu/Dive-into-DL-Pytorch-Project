import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = "../data"
fname = os.path.join(DATA_ROOT, "aclImdb_v1.tar.gz")
# 1 读取数据
if not os.path.exists(os.path.join(DATA_ROOT, "aclImdb")):
    print("从压缩包解压...")
    with tarfile.open(fname, 'r') as f:
        f.extractall(DATA_ROOT)

from tqdm import tqdm
import os
import random


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def read_imdb(folder='train', data_root="../data/aclImdb"):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


train_data, test_data = read_imdb('train'), read_imdb('test')


# 2 预处理理数据
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def get_tokenized_imdb(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review) for review, _ in data]


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])

    # 创建词汇表，添加特殊token
    vocab = Vocab.vocab(counter, min_freq=5)
    vocab.insert_token("<pad>", 0)  # 填充标记
    vocab.insert_token("<unk>", 1)  # 未知词标记
    vocab.set_default_index(vocab["<unk>"])  # 设置默认索引为<unk>

    return vocab


vocab = get_vocab_imdb(train_data)
print('# words in vocab:', len(vocab))


# 本函数已保存在d2lzh_torch包中方便以后使用
def preprocess_imdb(data, vocab):
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)

    # 使用vocab直接映射
    features = torch.tensor([pad([vocab[word] for word in words])
                             for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


# 3 创建数据迭代器
batch_size = 64
train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('#batches:', len(train_iter))


class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs


embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

# 加载GloVe词向量
print("正在加载GloVe词向量...")
glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(DATA_ROOT, "glove"))


# 本函数已保存在d2lzh_torch包中方便以后使用
def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])  # 初始化为0
    oov_count = 0  # out of vocabulary

    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1

    if oov_count > 0:
        print("There are %d oov words." % oov_count)

    return embed


# 获取词表
vocab_list = vocab.get_itos()  # 使用get_itos获取词汇列表
net.embedding.weight.data.copy_(load_pretrained_embedding(vocab_list, glove_vocab))
net.embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它

lr, num_epochs = 0.01, 5
# 要过滤掉不不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


# 本函数已保存在d2lzh_pytorch包中⽅方便便以后使⽤用
def predict_sentiment(net, vocab, sentence):
    """sentence是词语的列列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab[word] for word in sentence],
                            device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'