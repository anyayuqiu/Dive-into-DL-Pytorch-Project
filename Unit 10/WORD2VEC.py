import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data

sys.path.append("..")
print(torch.__version__)


# 处理数据集
with open('../data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

print('# sentences: %d' % len(raw_dataset)) # 输出 '# sentences: 42068'

# for st in raw_dataset[:3]:
#     print('# tokens:', len(st), st[:5])


# tk是token的缩写
# 保留数据集中出现5次以上的词
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

# 构建从索引到单词的映射列表：将counter中的每个单词（键）提取出来，组成一个列表
# 列表的索引即为该单词的ID，列表的顺序由counter.items()的顺序决定（通常按插入顺序，但Counter不保证顺序）
idx_to_token = [tk for tk, _ in counter.items()]

# 构建从单词到索引的映射字典：遍历idx_to_token列表，为每个单词分配一个唯一的整数ID
# 这样可以通过单词快速查找其对应的ID，用于后续的向量化表示
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}

# 将原始文本数据集转换为索引表示的数据集
# 对raw_dataset中的每个句子（st），将其中的每个单词（tk）转换为对应的ID
# 只转换那些存在于token_to_idx字典中的单词（即出现在词汇表中的单词）
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]

# 计算数据集中所有句子的总单词数（即所有索引的总数）
# 通过遍历dataset中的每个句子，计算其长度并求和
num_tokens = sum([len(st) for st in dataset])

def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens)
subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]

def compare_counts(token):
    return '# %s: before=%d, after=%d' % (
        token,
        sum([st.count(token_to_idx[token]) for st in dataset]),
        sum([st.count(token_to_idx[token]) for st in subsampled_dataset])
    )


# 初始化两个空列表，用于存储中心词和对应的背景词列表
centers, contexts = [], []

def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

# 创建一个微型数据集，包含两个句子：
# 第一个句子：0,1,2,3,4,5,6
# 第二个句子：7,8,9
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)

# 调用get_centers_and_contexts函数生成中心词和背景词
# max_window_size=2表示最大窗口大小为2
# 使用zip(*...)将返回的两个列表解包并配对
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)


all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index],
                self.contexts[index],
                self.negatives[index])

    def __len__(self):
        return len(self.centers)


def batchify(data):
    """
    用作DataLoader的参数collate_fn: 输入是个长度为batchsize的list,
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []

    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers.append(center)
        contexts_negatives.append(context + negative + [0] * (max_len - cur_len))
        masks.append([1] * cur_len + [0] * (max_len - cur_len))
        labels.append([1] * len(context) + [0] * (max_len - len(context)))

    return (torch.tensor(centers).view(-1, 1),
            torch.tensor(contexts_negatives),
            torch.tensor(masks),
            torch.tensor(labels))


# 参数设置
batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4

# 创建数据集
dataset = MyDataset(
    all_centers,
    all_contexts,
    all_negatives
)

# 创建数据加载器
data_iter = Data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=batchify,
    num_workers=num_workers
)

# 检查第一个批次的数据形状
for batch in data_iter:
    center_batch, context_neg_batch, mask_batch, label_batch = batch

    print(f"centers shape: {center_batch.shape}")
    print(f"contexts_negatives shape: {context_neg_batch.shape}")
    print(f"masks shape: {mask_batch.shape}")
    print(f"labels shape: {label_batch.shape}")

    break


# 词嵌入层
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(embed.weight)

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
print(embed(x))

X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
torch.bmm(X, Y).shape

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

# 训练模型
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):  # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", weight=mask
        )
        return res.mean(dim=1)


loss = SigmoidBinaryCrossEntropyLoss()

pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签变量量label中的1和0分别代表背景词和噪声词
label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]) # 掩码变量量
loss(pred, label, mask) * mask.shape[1] / mask.float().sum(dim=1)

def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print('%.4f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4))
# 注意1-sigmoid(x) = sigmoid(-x)
print('%.4f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))

#模型参数初始化
embed_size = 100
net = nn.Sequential(
 nn.Embedding(num_embeddings=len(idx_to_token),
embedding_dim=embed_size),
 nn.Embedding(num_embeddings=len(idx_to_token),
embedding_dim=embed_size)
)


def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])

            # 使用掩码变量mask来避免填充项对损失函数计算的影响
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean()  # 一个batch的平均loss

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1

        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))

train(net, 0.01, 10)