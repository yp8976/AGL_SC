# ndcg 0.685 recall 0.169337979
# from ast import If
# from pyexpat import model
import numpy as np
import os
# import torch

# from torch import nn
import math

import mindspore
from mindspore.common.initializer import initializer, Normal
from mindspore import ops, nn
from sklearn import preprocessing
import numpy as np
import evaluating_indicator

from tqdm import tqdm
import os
import time


def read_dataset(filename):
    orgin_data = []
    with open(filename, encoding="UTF-8") as fin:  # 取出训练数据,并切分为用户，物品，评分 数据
        line = fin.readline()
        while line:
            user, item, rating = line.strip().split("\t")
            orgin_data.append((int(user)-1, int(item)-1, float(rating)))
            line = fin.readline()  # 读取一行数据

    user, item = set(), set()
    for u, v, r in orgin_data:
        user.add(u)
        item.add(v)
    user_list = list(user)
    item_list = list(item)
    uLen = max(user_list)+1
    vLen = max(item_list)+1
    # orgin_data是原始数据，positive用到了，考虑要不要留着,不一定管事
    return orgin_data, user_list, item_list, uLen, vLen


def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1
    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])

    return train_data, test_data, n_user, m_item


def read_data(train_file, test_file):
    train_data = []
    test_data = []
    with open(train_file, encoding="UTF-8") as fin:  # 取出训练数据
        line = fin.readline()
        while line:
            user, item, rating = line.strip().split("\t")
            train_data.append((int(user)-1, int(item)-1, float(rating)))
            line = fin.readline()  # 读取一行数据
    user, item, _ = zip(*train_data)
    num_users = max(user) + 1
    num_items = max(item) + 1
    with open(test_file, encoding="UTF-8") as fin:  # 取出训练数据
        line = fin.readline()
        while line:
            user, item, rating = line.strip().split("\t")
            test_data.append((int(user)-1, int(item)-1, float(rating)))
            line = fin.readline()  # 读取一行数据
    user, item, _ = zip(*test_data)
    num_users = max(max(user) + 1, num_users)
    num_items = max(max(item) + 1, num_items)

    return train_data, num_users, num_items


def create_RMN(train_data, uLen, vLen):  # 用总的构建RMN然后切分？在lightgcn中使用随机种子切分
    data = []
    W = mindspore.numpy.zeros((uLen, vLen))  # 先建一个m*n的矩阵
    indices = mindspore.numpy.array([[u, v] for u, v in train_data], dtype=mindspore.int32)
    values = mindspore.numpy.ones(len(train_data), dtype=mindspore.float32)

    # 使用tensor_scatter_update进行赋值
    W = ops.tensor_scatter_update(W, indices, values)
    # for w in tqdm(train_data):
    #     u, v = w
    #     data.append((u, v))
    #     W[u][v] = 1   # 使用索引建立W矩阵

    return W


def sumpow(x, k):    # 计算次方求和
    sum = 0
    for i in range(k+1):
        sum += math.pow(x, i)
    return sum


def lightgcn_init(user_path, item_path):
    print("初始化向量")
    # 判断是否有初始向量
    # "init-vectors_u.dat","init-vectors_v.dat"

    if os.access(user_path, os.F_OK) and os.access(item_path, os.F_OK):
        print("正在加载初始向量")
        # 若存在 则使用
        u_vector = mindspore.load_checkpoint(user_path)
        v_vector = mindspore.load_checkpoint(item_path)

        # print(u_vector)
        u_vectors = u_vector
        # u_vectors = u_vectors.to(device)
        u_vectors = mindspore.Parameter(u_vectors)
        # print(u_vectors)
        v_vectors = v_vector
        # v_vectors = v_vectors.to(device)
        v_vectors = mindspore.Parameter(v_vectors)
    else:
        raise Exception("没有找到初始向量")
    return u_vectors, v_vectors


def computeResult(net, test_data):
    node_list_u_, node_list_v_ = {}, {}
    test_user, test_item, test_rate = test_data
    i = 0
    for item in net.u.weight:
        node_list_u_[i] = {}
        node_list_u_[i]['embedding_vectors'] = item.cpu().detach().numpy()
        i += 1

    # 对于v 需要在这里映射一下
    i = 0
    for item in net.v.weight:
        node_list_v_[i] = {}
        node_list_v_[i]['embedding_vectors'] = item.cpu().detach().numpy()
        i += 1
#     test_user, test_item, test_rate = read_data(r"../data/dblp/rating_test.dat")
    # 目标:recommendation metrics: F1 : 0.1132, MAP : 0.2041, MRR : 0.3331, NDCG : 0.2609
    # F1：0.1137  MAP：0.1906 MRR： 0.3336 NDCG：0.2619
    # 对初始向量计算f1值,map值,mrr值,mndcg值
    f1, map, mrr, mndcg = evaluating_indicator.top_N(
        net, test_user, test_item, test_rate, node_list_u_, node_list_v_, top_n=10)
    print("f1:", f1, "map:", map, "mrr:", mrr, "mndcg:", mndcg)
    return mndcg


def utils2(x, y, p=2):
    wasserstein_distance = ops.abs(
        (
            ops.sort(x.swapaxes(0, 1), axis=1)[0]
            - ops.sort(y.swapaxes(0, 1), axis=1)[0]
        )
    )
    wasserstein_distance = ops.pow(
        ops.sum(ops.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    wasserstein_distance = ops.pow(
        ops.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance


def init_vectors(rank, uLen, vLen):  # 初始化向量
    # 初始向量正则化
    # 随机初始化向量和UV分解向量 都固定下来
    print("初始化向量")

    # u_vectors = preprocessing.normalize(vectors_u, norm='l2')# 再转成tensor
    # u_vectors = mindspore.numpy.empty(uLen, rank)
    # 使用 Kaiming 初始化
    u_vectors = np.random.random([uLen, rank])
    mindspore.common.initializer.HeNormal(u_vectors, nonlinearity='tanh')
    u_vectors = preprocessing.normalize(u_vectors, norm='l2')

    # # 再转成tensor
    # v_vectors = mindspore.numpy.empty(vLen, rank)
    # 使用 Kaiming 初始化
    v_vectors = np.random.random([vLen, rank])
    mindspore.common.initializer.HeNormal(v_vectors, nonlinearity='tanh')
    v_vectors = preprocessing.normalize(v_vectors, norm='l2')

    return u_vectors, v_vectors


def gpu():
    from pynvml import (nvmlInit,
                        nvmlDeviceGetCount,
                        nvmlDeviceGetHandleByIndex,
                        nvmlDeviceGetMemoryInfo,
                        nvmlShutdown)

    # 初始化
    nvmlInit()

    # 获取设备数量
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        # 获取设备
        handle = nvmlDeviceGetHandleByIndex(i)
        # 获取设备信息
        info = nvmlDeviceGetMemoryInfo(handle)

        print(f"Device {i}: ")
        print(f"Total memory: {info.total/1024**2} MB")
        print(f"Free memory: {info.free/1024**2} MB")
        print(f"Used memory: {info.used/1024**2} MB")

    # 清理
    nvmlShutdown()


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):
    # x, params['item_num'], params['negative_num'=200~800], interacted_items, params['sampling_sift_pos'=no]
    neg_candidates = np.arange(item_num)
    if sampling_sift_pos:
        neg_items = []
        for u in pos_train_data[0]:
            probs = np.ones(item_num)
            probs[interacted_items[u]] = 0
            probs /= np.sum(probs)

            u_neg_items = np.random.choice(
                neg_candidates, size=neg_ratio, p=probs, replace=True).reshape(1, -1)

            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(
            neg_candidates, (len(pos_train_data[0]), neg_ratio), replace=True)

    neg_items = mindspore.Tensor.from_numpy(neg_items)

    # users, pos_items, neg_items
    return pos_train_data[0], pos_train_data[1], neg_items


class GRU_Cell(nn.Cell):

    def __init__(self, in_dim, hidden_dim):
        '''
        :param in_dim: 输入向量的维度
        :param hidden_dim: 输出的隐藏层维度
        '''
        super(GRU_Cell, self).__init__()
        self.rx_linear = nn.Dense(in_dim, hidden_dim)
        self.rh_linear = nn.Dense(hidden_dim, hidden_dim)
        self.zx_linear = nn.Dense(in_dim, hidden_dim)
        self.zh_linear = nn.Dense(hidden_dim, hidden_dim)
        self.hx_linear = nn.Dense(in_dim, hidden_dim)
        self.hh_linear = nn.Dense(hidden_dim, hidden_dim)

    def construct(self, x, h_1):
        '''
        :param x:  输入的序列中第t个物品向量 [ batch_size, in_dim ]
        :param h_1:  上一个GRU单元输出的隐藏向量 [ batch_size, hidden_dim ]]
        :return: h 当前层输出的隐藏向量 [ batch_size, hidden_dim ]
        '''

        r = ops.sigmoid(self.rx_linear(x)+self.rh_linear(h_1))
        z = ops.sigmoid(self.zx_linear(x)+self.zh_linear(h_1))
        h_ = ops.tanh(self.hx_linear(x)+self.hh_linear(r*h_1))
        h = z*h_1+(1-z)*h_
        return h


class Net(nn.Cell):
    # u_len用户数量，v_len物品数量，u_vectors用户向量，v_vectors物品向量
    def __init__(self, config, u_vectors, v_vectors, q_dims=None, dropout=0.5):
        super(Net, self).__init__()
        u_len = config['ulen']
        v_len = config['vlen']
        rank = config['rank']
        self.u = nn.Embedding(u_len, rank)
        self.v = nn.Embedding(v_len, rank)

        # vae
        p_dims = config["p_dims"]
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1]]
        self.q_layers = nn.CellList([nn.Dense(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.CellList([nn.Dense(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.update = 0
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.drop = nn.Dropout(dropout)

        self.mlplist = nn.CellList(
            [nn.SequentialCell(nn.Dense(rank, rank*2), nn.ReLU(), nn.Dense(2*rank, rank))])
        self.vaelist = nn.CellList(
            [nn.SequentialCell(nn.Dense(v_len, 600), nn.Dense(600, 400))])
        # self.edge_mask_learner = nn.CellList([nn.Sequential(nn.Linear(2 * rank, rank), nn.ReLU(), nn.Linear(rank, 1))])
        # self.lstm = nn.LSTM(input_size=1, hidden_size=200, batch_first=True)
        self.GRU = nn.GRU(input_size=1, hidden_size=200, batch_first=True)

        # GRU
        self.rnn_cell = GRU_Cell(in_dim=1, hidden_dim=400)
        self.init_weights()

    def gru(self, x, h=None):
        # x 的形状现在是 [batch_size, seq_len, in_dim]
        if h is None:
            h = ops.zeros(x.shape[0], self.hidden_dim)  # 初始化隐藏状态

        outs = []
        # 遍历每一个时间步
        for t in range(x.shape[1]):  # 现在 x.shape[1] 是 seq_len
            seq_x = x[:, t, :]  # 取出所有 batch 的第 t 个时间步
            h = self.rnn_cell(seq_x, h)
            outs.append(ops.unsqueeze(h, 1))

        outs = ops.cat(outs, axis=1)  # 将输出连接起来
        return outs, h

    def nfm1(self):
        # 计算二阶交互项，使用FM的思想
        interaction = ops.mm(self.u.weight, self.v.weight.t())

        # 特征交互池化层
        square_of_sum = ops.pow(ops.sum(interaction, dim=1), 2)
        sum_of_squares = ops.sum(ops.pow(interaction, 2), dim=1)

        square_of_sum1 = ops.pow(ops.sum(interaction, dim=0), 2)
        sum_of_squares1 = ops.sum(ops.pow(interaction, 2), dim=0)

        # 计算最终的交互项
        inter_term = 0.5 * (square_of_sum - sum_of_squares).detach().cpu()
        inter_term = inter_term.view(-1, 1)
        # print(inter_term.shape)
        inter_term = nn.Dense(1, 128)(inter_term)

        inter_term1 = 0.5 * (square_of_sum1 - sum_of_squares1).detach().cpu()
        inter_term1 = inter_term1.view(-1, 1)
        # print(inter_term1.shape)
        inter_term1 = nn.Dense(1, 128)(inter_term1)
        # combined_embedding = torch.cat((inter_term, inter_term1), dim=0)
        # inter_term = torch.mm(inter_term1,inter_term.t())
        # print(inter_term)
        # inter_term = inter_linear(linear_layer)
        # 神经网络部分
        # nn_output = F.relu(self.fc1(inter_term))
        # nn_output = self.fc2(nn_output)
        # nn_output = self.mlp(inter_term)
        # self.fc_layers = nn.CellList()
        # hidden_dims = [256,128]
        # for idx, (in_size, out_size) in enumerate(zip([rank] + hidden_dims[:-1], hidden_dims)):
        #     self.fc_layers.append(nn.Linear(in_size, out_size))
        #     self.fc_layers.append(nn.ReLU())
        #     # if dropout_rate > 0:
        #     #     self.fc_layers.append(nn.Dropout(dropout_rate))

        # self.output_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        # x = inter_term
        # x1 = inter_term1
        # x = torch.mm(x,x1.t())
        x = self.mlp(inter_term)
        x1 = self.mlp(inter_term1)
        x = ops.mm(x, x1.t())
        # for layer in self.mlplist:
        #     x = layer(x.to(device))
        # x1 = layer(x1.to(device))
        # print("111",x.shape,x1.shape)
        # out = self.output_layer(x)
        # out = torch.mm(x,x1.t())

        # 总输出，这里可以使用sigmoid函数进行预测

        return x

    def nfm2(self):
        # 计算二阶交互项，使用FM的思想
        interaction = ops.mm(self.u.weight, self.v.weight.t())

        # 特征交互池化层
        square_of_sum = ops.pow(ops.sum(interaction, dim=1), 2)
        sum_of_squares = ops.sum(ops.pow(interaction, 2), dim=1)

        self.fc_layers = nn.CellList()
        hidden_dims = [128, 64]
        for idx, (in_size, out_size) in enumerate(zip([128] + hidden_dims[:-1], hidden_dims)):
            self.fc_layers.append(nn.Dense(in_size, out_size))
            self.fc_layers.append(nn.ReLU())
            # if dropout_rate > 0:
            #     self.fc_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Dense(hidden_dims[-1], 1)
        # 计算最终的交互项
        inter_term = 0.5 * (square_of_sum - sum_of_squares)
        x = inter_term.detach().cpu()
        print(x)
        for layer in self.fc_layers:
            x = layer(x)
        out = self.output_layer(x)
        out = ops.sigmoid(out)
        # print(inter_term.shape[0])
        # inter_term  = nn.Linear(inter_term.shape[0], 128)(inter_term).to(device)
        # print(inter_term)
        # # inter_term = inter_linear(linear_layer)
        # # print(inter_term.shape)
        # # 神经网络部分
        # # nn_output = F.relu(self.fc1(inter_term))
        # # nn_output = self.fc2(nn_output)
        # nn_output = self.mlp(inter_term)

        # 总输出，这里可以使用sigmoid函数进行预测
        output = ops.sigmoid(out)

        return output

    def nfm(self):
        # 计算二阶交互项，使用FM的思想
        interaction = ops.mm(self.u.embedding_table, self.v.embedding_table.t())

        # 特征交互池化层
        square_of_sum = ops.pow(interaction, 2)

        self.fc_layers = nn.CellList()
        hidden_dims = [600, 400, 200]
        for idx, (in_size, out_size) in enumerate(zip([vLen] + hidden_dims[:-1], hidden_dims)):
            self.fc_layers.append(nn.Dense(in_size, out_size))
            self.fc_layers.append(nn.ReLU())
            # if dropout_rate > 0:
            #     self.fc_layers.append(nn.Dropout(dropout_rate))

        # self.output_layer = nn.Linear(hidden_dims[-1], 400).to(device)
        # 计算最终的交互项
        inter_term = 0.5 * (square_of_sum)
        for layer in self.fc_layers:
            inter_term = layer(inter_term)
        output = inter_term
        # out = self.output_layer(inter_term)
        # output = torch.sigmoid(out)
        # print(inter_term.shape[0])
        # inter_term  = nn.Linear(inter_term.shape[0], 128)(inter_term).to(device)
        # print(inter_term)
        # # inter_term = inter_linear(linear_layer)
        # # print(inter_term.shape)
        # # 神经网络部分
        # # nn_output = F.relu(self.fc1(inter_term))
        # # nn_output = self.fc2(nn_output)
        # nn_output = self.mlp(inter_term)

        # 总输出，这里可以使用sigmoid函数进行预测
        # output = torch.sigmoid(out)

        return output

    def encode(self, input):
        L2 = ops.L2Normalize()
        h = L2(input)
        h = self.drop(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = ops.tanh(h)
            else:
                W = self.nfm()
                W = W.unsqueeze(0)
                h = h.unsqueeze(-1)
                # output torch.Size([1435, 1522, 400]) h torch.Size([1, 1435, 400])
                output, h = self.GRU(h, W)
                # W = input.unsqueeze(-1)
                # h = h.unsqueeze(0)
                # output, h = self.GRU(W, h)
                h1 = h.squeeze(0)
                h2 = output.sum(axis=1)
                mu = h1
                logvar = h2

            # W = input.unsqueeze(-1)       #这段内存不够了
            # h = h.unsqueeze(0)
            # output, h = self.gru(W, h)
            # h = h.squeeze(0)
            # mu = h[:, :self.q_dims[-1]]  # 在这里添加gru模块试试，
            # logvar = h[:, self.q_dims[-1]:]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = ops.exp(0.5 * logvar)
            # eps = self.mlpvae(input)
            eps = ops.rand_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = ops.tanh(h)
        return h

    def init_weights_origin(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.shape
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            # layer.weight.set_data(initializer(Normal(
            #     sigma=std, mean=0.0)), layer.weight.shape, layer.weight.dtype)

            # Normal Initialization for Biases
            # layer.bias.set_data(initializer(Normal(
            #     sigma=0.001, mean=0.0)), layer.bias.shape, layer.bias.dtype)


        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.shape
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            # layer.weight.set_data(initializer(Normal(
            #     sigma=std, mean=0.0)), layer.weight.shape, layer.weight.dtype)

            # # Normal Initialization for Biases
            # layer.bias.set_data(initializer(Normal(
            #     sigma=0.001, mean=0.0)), layer.bias.shape, layer.bias.dtype)

    def init_weights(self):
        mindspore.common.initializer.HeNormal(self.u.embedding_table, nonlinearity='tanh') 
        mindspore.common.initializer.HeNormal(self.v.embedding_table, nonlinearity='tanh') 
        for layer in self.q_layers:
            # Kaiming Initialization for weights
            mindspore.common.initializer.HeNormal(layer.weight.data, nonlinearity='tanh')

            # Normal Initialization for Biases
            mindspore.common.initializer.HeNormal(layer.bias.data, nonlinearity='tanh')
            # layer.bias.set_data(initializer(Normal(
            #     sigma=0.001, mean=0.0)), layer.bias.shape, layer.bias.dtype)

        for layer in self.p_layers:
            # Kaiming Initialization for weights
            mindspore.common.initializer.HeNormal(layer.weight.data, nonlinearity='tanh')

            # Normal Initialization for Biases
            mindspore.common.initializer.HeNormal(layer.bias.data, nonlinearity='tanh')
            # layer.bias.set_data(initializer(Normal(
            #     sigma=0.001, mean=0.0)), layer.bias.shape, layer.bias.dtype)

    def init_weights_x(self):
        for layer in self.q_layers:
            # Kaiming Initialization for weights
            torch.nn.init.xavier_normal_(layer.weight.data)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Kaiming Initialization for weights
            torch.nn.init.xavier_normal_(layer.weight.data)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def vae(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 *
                         self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        kl_loss = (
            -0.5
            * ops.mean(ops.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        ce_loss = -(ops.log_softmax(z, 1) * input).sum(1).mean()

        return kl_loss+ce_loss, z

    def mlp(self):
        combined_embedding = ops.cat((self.u.weight, self.v.weight), axis=0)
        for layer in self.mlplist:
            combined_embedding = layer(combined_embedding)
        user_mask, item_mask = ops.split(
            combined_embedding, [uLen, vLen], axis=0)
        # user_mask.requires_grad = True
        # item_mask.requires_grad = True
        # self.u.weight = nn.Parameter(user_mask)
        # self.v.weight = nn.Parameter(item_mask)
        return ops.mm(user_mask, item_mask.t())

    def mlpvae(self, RR):
        L2 = ops.L2Normalize()
        h = L2(RR)
        h = self.drop(h)
        for i, layer in enumerate(self.vaelist):
            h = layer(h)
            if i != len(self.vaelist) - 1:
                h = ops.tanh(h)
        return h

    def construct(self, W):
        RR = ops.mm(self.u.embedding_table, ops.t(self.v.embedding_table))  # 随着UV的更新R也更新，即R'
        logp_R = ops.log_softmax(RR, axis=-1)  # [6001, 6001]
        p_R = ops.softmax(W, axis=-1)  # [6001, 6001]
        kl_sum_R = ops.KLDivLoss(reduction='sum')(logp_R, p_R)
        C = utils2(W, RR)
        loss = kl_sum_R+0.0005*C
        return loss, RR

    def construct1(self, W):
        RR = ops.mm(self.u.embedding_table, ops.t(self.v.embedding_table))  # 随着UV的更新R也更新，即R'
        # RR = self.mlp()
        logp_R = ops.log_softmax(RR, axis=-1)  # [6001, 6001]
        tenW = W
        # tenW = F.logsigmoid(tenW)
        p_R = ops.softmax(tenW, axis=-1)  # [6001, 6001]
        kl_sum_R = nn.KLDivLoss(reduction='sum')(logp_R, p_R)
        # RR = self.mlp()
        # C = utils2(W.to(device), RR)
        RR = self.mlp()
        # z,loss2 = self.vae(W)
        logp_R = ops.log_softmax(W, axis=-1)
        p_R = ops.softmax(RR, axis=-1)
        loss3 = nn.KLDivLoss(reduction='sum')(logp_R, p_R)  # 0.1856
        # print("klR,loss2.loss3",kl_sum_R.item(),loss3.item())
        return loss3+0.1*kl_sum_R, RR


def train_mlp(config, model, optimizer, W, W2, k):
    best = 0.0
    best2 = 0.0
    bestR = []
    bestE = 0
    converged_epochs = 0
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        should_stop = False
        for epoch in range(config['epochs']):
            start_time = time.time()
            model.set_train(True)
            R, loss = model.construct1(W)
            # R = makeup(R.detach().cpu())
            # R = model.mlp()
            # z,loss2 = model.vae(R.detach().cpu().to(device))

            loss.backward()
            optimizer.step()
            # n20, r20 = evaluate(model,test_data_tr, test_data_te)
            # n20 = evaluate2(test_data_tr, R, k=20)
            model.set_train(False)
            # val = eval(model.u.weight, model.v.weight)
            ndcg = multivae.eval_ndcg(model, W2, R, k=k)
            recall = multivae.eval_recall(model, W2, R, k=k)
            print("|Epoch", epoch, "|NDCG:", ndcg,
                  "|Recall:", recall, '|Loss', loss.item())
            converged_epochs += 1
            if ndcg > best:
                best = ndcg
                bestE = epoch
                bestR = R
                converged_epochs = 0
            if converged_epochs >= 10 and epoch > 50:
                print('模型收敛，停止训练。最优ndcg值为：', best,
                      "最优epoch为：\n", bestE, "最优R为：\n", bestR)
                # torch.save(net.state_dict(), 'model_parameters.pth')
                print("保存模型参数")
                # torch.save(net.u.weight, 'u_parameters.pth')
                # torch.save(net.v.weight, 'v_parameters.pth')
                break
            if epoch == config['epochs'] - 1:
                print('模型收敛，停止训练。最优ndcg值为：', best, "最优R为：\n", bestR)
            if should_stop:
                break
        # sys.exit()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print('=' * 50)
    train_time = time.strftime(
        "%H: %M: %S", time.gmtime(time.time() - start_time))
    print("训练时间:", train_time)
    print('| 训练结束 | 最优轮:{0}|最优 NDCG@{1}:{2}'.format(bestE,
          config['topk'], best))
    print('=' * 50)

    return bestR

def train_gnn(config, model, optimizer, W, W2):
    best = 0.0
    bestR = []
    bestE = 0
    converged_epochs = 0
    # At any point you can hit Ctrl + C to break out of training early.
    def forward_fn(W1):
        loss, RR = model(W1)
        return loss, RR
    
    try:
        should_stop = False
        for epoch in range(config['epochs']):
            start_time = time.time()
            model.set_train()
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, R), grads = grad_fn(W)
            optimizer(grads)

            model.set_train(False)
            # for param in model.trainable_params():
            #     print(param.name, param.asnumpy().sum())
            # val = eval(model.u.weight, model.v.weight)
            # print("|Epoch", epoch, "|Val", val)
            # val = val['ndcg']
            from multivae1 import eval_ndcg, eval_recall, eval_precision
            val = eval_ndcg(model, W2, R, k=20)
            print("NDCG:",val)
            converged_epochs += 1
            if val > best:
                best = val
                bestE = epoch
                bestR = R
                converged_epochs = 0
            if converged_epochs >= 10 and epoch > 100:
                print('模型收敛，停止训练。最优ndcg值为：', best,
                      "最优epoch为：\n", bestE, "最优R为：\n", bestR)
                # torch.save(net.state_dict(), 'model_parameters.pth')
                print("保存模型参数")
                # torch.save(net.u.weight, 'u_parameters.pth')
                # torch.save(net.v.weight, 'v_parameters.pth')
                break
            if epoch == config['epochs'] - 1:
                print('模型收敛，停止训练。最优ndcg值为：', best, "最优R为：\n", bestR)
            if should_stop:
                break
        # sys.exit()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print('=' * 50)
    train_time = time.strftime(
        "%H: %M: %S", time.gmtime(time.time() - start_time))
    print("训练时间:", train_time)
    print('| 训练结束 | 最优轮:{0}|最优 NDCG@{1}:{2}'.format(bestE,
          config['topk'], best))
    print('=' * 50)

    return bestR


def train_vae(config, model, optimizer, W, W2):
    best = 0.0
    bestR = []
    bestE = 0
    converged_epochs = 0
    def forward_fn(W1):
        loss, z = model.vae(W1)
        return loss, z
    # At any point you can hit Ctrl + C to break out of training early.

    try:
        should_stop = False
        for epoch in range(config['epochs']):
            start_time = time.time()
            model.set_train()
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, R), grads = grad_fn(W)
            optimizer(grads)
            model.set_train(False)
            from multivae1 import eval_ndcg, eval_recall, eval_precision
            val = eval_ndcg(model, W2, R, k=40)
            val_r = eval_recall(model, W2, R, k=40)
            percison = eval_precision(model, W2, R, k=20)

            print('=' * 89)
            print('| Epoch {:2d}|loss {:4.5f} | percision {:4.5f} | r20 {:4.5f}| n20 {:4.5f} |'.format(
                epoch, loss.item(), percison, val_r, val))
            converged_epochs += 1
            if val_r > best:
                best = val_r
                bestE = epoch
                bestR = R
                converged_epochs = 0
            if converged_epochs >= 10 and epoch > 100:
                print('模型收敛，停止训练。最优ndcg值为：', best,
                      "最优epoch为：\n", bestE, "最优R为：\n", bestR)
                # torch.save(net.state_dict(), 'model_parameters.pth')
                print("保存模型参数")
                # torch.save(net.u.weight, 'u_parameters.pth')
                # torch.save(net.v.weight, 'v_parameters.pth')
                break
            # if loss < 0.0001:
            #     break
            if epoch == config['epochs'] - 1:
                print('模型收敛，停止训练。最优ndcg值为：', best, "最优R为：\n", bestR)
            if should_stop:
                break
        # sys.exit()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    print('=' * 50)
    train_time = time.strftime(
        "%H: %M: %S", time.gmtime(time.time() - start_time))
    print("训练时间:", train_time)
    print('| 训练结束 | 最优轮:{0}|最优 NDCG@{1}:{2}'.format(bestE,
          config['topk'], best))
    print('=' * 50)

    return bestR



config = {
    # AmazonElectronics 5e-4,100K 1e-3,Movielens1M 1e-3
    'dataset': 'AmazonElectronics',
    'topk': 40,
    'lr': 5e-4,
    'wd': 0.0,
    'rank': 128,
    'batch_size': 512,
    'testbatch': 100,
    'epochs': 1000,
    'total_anneal_steps': 200000,
    'anneal_cap': 0.2,
    'seed': 2020,
}

if __name__ == "__main__":
    from evalu import run  # 导入评估
    import multivae
    import os
    mindspore.set_context(device_target='GPU', device_id=0)
    print('Random seed: {}'.format(config['seed']))
    np.random.seed(config['seed'])
    mindspore.dataset.config.set_seed(config['seed'])

    eval = run(config['dataset'])
    # print("公式：loss = C+kl_sum_R") #[0.21845276]
    rank = 128  # 向量维度
    epochs = config['epochs']  # 迭代次数
    lr = config['lr']  # 学习率 这里的学习率和vae的要区分开
    iters = 1  # 每x轮输出一次
    top_n = config['topk']  # topn推荐
    dataset = config['dataset']
    batch_size = config['batch_size']
    test_batch_size = config['testbatch']

    print("dataset", dataset, "rank:", rank, "epochs:",
          epochs, "lr:", lr, "topK@", top_n, "iters:", iters)
    path = os.path.dirname(os.path.dirname(__file__))
    train_file = r"../data/"+dataset+"/train.txt"  # 训练集路径
    test_file = r"../data/"+dataset+"/test.txt"
    print("train_file:", train_file)
    print("test_file:", test_file)
    train_data, test_data, uLen, vLen = load_data(train_file, test_file)
    config['ulen'] = uLen
    config['vlen'] = vLen
    config['n_items'] = vLen
    p_dims = [200, 600, vLen]
    config['p_dims'] = p_dims
    print("用户项目数：", uLen, vLen)
    print("创建初始W,W2矩阵")
    W = create_RMN(train_data, uLen, vLen)
    W2 = create_RMN(test_data, uLen, vLen)

    m = None
    n = None

    replaced_indices = []

    # lightgcn_init
    # user_path = r"/home/yuanpeng/project/b/code/u_0.pt"
    # item_path = r"/home/yuanpeng/project/b/code/v_0.pt"
    # u_vectors, v_vectors = lightgcn_init(user_path,item_path)

    # kaiming_init
    u_vector, v_vector = init_vectors(rank, uLen, vLen)
    u_vectors = mindspore.Tensor.from_numpy(u_vector).astype(mindspore.float32)
    v_vectors = mindspore.Tensor.from_numpy(v_vector).astype(mindspore.float32)
    # u_vectors = mindspore.Parameter(u_vectors)
    # v_vectors = mindspore.Parameter(v_vectors)

    model = Net(config, u_vectors, v_vectors)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.0001)

    R_fake_path = r'R_'+dataset+'.ckpt'
    R_fake_np = r'R_'+dataset+'.npy'
    z_path = r'R_vae_'+dataset+'.ckpt'
    z_np = r'R_vae_'+dataset+'.npy'
    R_fake = W
    z = W

    if os.path.exists(R_fake_path) and os.path.exists(R_fake_np):
        param_dict = mindspore.load_checkpoint(R_fake_path)
    
        # 为模型加载参数
        model_params = {name: param for name, param in param_dict.items() if "model." in name}
        mindspore.load_param_into_net(model, model_params)
        optimizer_params = {name: param for name, param in param_dict.items() if "optimizer." in name}
        mindspore.load_param_into_net(optimizer, optimizer_params)

        np_array = np.load(R_fake_np)
        R_fake = mindspore.Tensor(np_array)

        print("载入训练矩阵R.")

    else:
        # 如果文件不存在，进行训练并保存矩阵R
        print("重新训练矩阵R.")
        R_fake = train_gnn(config, model, optimizer, R_fake, W2)
        model_params = {f"model.{name}": param for name, param in model.parameters_and_names()}
        optimizer_params = {f"optimizer.{name}": param for name, param in optimizer.parameters_and_names()}
        save_obj = {**model_params, **optimizer_params}
        mindspore.save_checkpoint(save_obj, R_fake_path)
        np.save(R_fake_np, R_fake.asnumpy())

    print("重新训练矩阵z.")

    z = train_vae(config, model, optimizer, R_fake, W2)