# 如果没有安装pandas，则反注释下⾯面⼀一⾏行行
# !pip install pandas

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

#2 获取数据集
#训练集有 1460个样本 80个特征 1个标签
train_data = pd.read_csv('data/train.csv')
#测试集有 1459个样本 80个特征
test_data = pd.read_csv('data/test.csv')
# print(train_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

#去除第一个id 和 最后的标签 张量形状(2919, 79)
all_features = pd.concat((train_data.iloc[:, 1:-1],test_data.iloc[:, 1:]))

#print(all_features.shape)
#3 数据预处理

#取出类型不为"object"的列
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
#进行标准化 减去均值 除以标准差
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

# 标准化后，每个特征的均值变为0，所以可以直接⽤用0来替换缺失值
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征

#补全缺失值
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
all_features = all_features.fillna(0)
all_features = all_features.astype(float)

n_train = train_data.shape[0]
# 将数据移动到GPU
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float).to(device)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float).to(device)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1).to(device)

#均方差（y - yy） ^ 2
loss = torch.nn.MSELoss()

#4 训练模型
def get_net(feature_num):
    net = nn.Linear(feature_num,1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    # 将模型移动到GPU
    return net.to(device)

def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0).to(device))
        rmse = torch.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    #这⾥里里使⽤用了了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 确保数据在正确的设备上
            X, y = X.to(device), y.to(device)
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

#5
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    #进行整除得到每一折的样本量
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

PATH = "./best_net.pt"
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    # 保留最佳模型
    best_net = None  # 在K折循环中保存最佳模型
    best_valid_rmse = float('inf')
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # 确保数据在GPU上
        data = [d.to(device) for d in data]
        #初始化网络，输入网络形状
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        if valid_ls[-1] < best_valid_rmse:
            best_valid_rmse = valid_ls[-1]
            best_net = net  # 保存当前最佳模型
            torch.save(net.state_dict(), PATH)

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k, best_net

k, num_epochs, lr, weight_decay, batch_size = 5, 400, 5, 0.1, 128
train_l, valid_l, best_net = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    # 将预测结果移回CPU进行后续处理
    preds = net(test_features).detach().cpu().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)

def train_and_pred_bynet(net, train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    # 将预测结果移回CPU进行后续处理
    preds = net(test_features).detach().cpu().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)

train_and_pred_bynet(best_net, train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)