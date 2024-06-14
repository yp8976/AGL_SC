"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""

import world

from dataloader import BasicDataset

import mindspore
from mindspore import ops, nn
from mindspore.common.initializer import Normal, initializer


class BasicModel(nn.Cell):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config["latent_dim_rec"]
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = ops.sum(users_emb * pos_emb, dim=1)
        neg_scores = ops.sum(users_emb * neg_emb, dim=1)
        loss = ops.mean(ops.softplus(neg_scores - pos_scores))
        reg_loss = (
            (1 / 2)
            * (
                users_emb.norm(2).pow(2)
                + pos_emb.norm(2).pow(2)
                + neg_emb.norm(2).pow(2)
            )
            / float(len(users))
        )
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


def utils2(x, y, p=2):
    wasserstein_distance = torch.abs(
        (
            torch.sort(x.transpose(0, 1), dim=1)[0]
            - torch.sort(y.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.pow(
        torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p
    )
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset, q_dims=None, dropout=0.5):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

        # vae
        p_dims = [200, 600, self.dataset.m_items]
        self.p_dims = p_dims
        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.CellList(
            [
                nn.Dense(d_in, d_out)
                for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])
            ]
        )
        self.p_layers = nn.CellList(
            [
                nn.Dense(d_in, d_out)
                for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
            ]
        )

        self.update = 0
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.mlplist = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(128, 128 * 2), nn.ReLU(), nn.Dense(2 * 128, 128)
                )
            ]
        )

    def init_weights(self):
        for layer in self.q_layers:
            # Kaiming Initialization for weights
            mindspore.common.initializer.HeNormal(
                layer.weight.data, nonlinearity="tanh"
            )

            # Normal Initialization for Biases
            layer.bias.data.set_data(
                initializer("normal", layer.bias.data.shape, layer.bias.data.dtype)
            )

        for layer in self.p_layers:
            # Kaiming Initialization for weights
            mindspore.common.initializer.HeNormal(
                layer.weight.data, nonlinearity="tanh"
            )

            # Normal Initialization for Biases
            layer.bias.data.set_data(
                initializer("normal", layer.bias.data.shape, layer.bias.data.dtype)
            )

    def mlp(self):
        uLen = self.dataset.n_users
        vLen = self.dataset.m_items
        combined_embedding = ops.cat(
            (self.embedding_user.weight, self.embedding_item.weight), dim=0
        )
        for layer in self.mlplist:
            combined_embedding = layer(combined_embedding)
        user_mask, item_mask = ops.split(combined_embedding, [uLen, vLen], axis=0)
        # user_mask.requires_grad = True
        # item_mask.requires_grad = True
        # self.u.weight = nn.Parameter(user_mask)
        # self.v.weight = nn.Parameter(item_mask)
        return ops.mm(user_mask, item_mask.t())

    def encode(self, input):
        L2 = ops.L2Normalize()
        h = L2(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = ops.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = ops.tanh(h)
        return h

    def vae(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        ce_loss = -(ops.log_softmax(z, 1) * input).sum(1).mean()

        return z, kl_loss + ce_loss

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lightGCN_n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["A_split"]
        self.embedding_user = nn.Embedding(
            vocab_size=self.num_users, embedding_size=self.latent_dim
        )
        self.embedding_item = nn.Embedding(
            vocab_size=self.num_items, embedding_size=self.latent_dim
        )
        if self.config["pretrain"] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function

            self.embedding_user.embedding_table.set_data(
                initializer(
                    "normal",
                    self.embedding_user.embedding_table.shape,
                    self.embedding_user.embedding_table.dtype,
                )
            )
            self.embedding_item.embedding_table.set_data(
                initializer(
                    "normal",
                    self.embedding_item.embedding_table.shape,
                    self.embedding_item.embedding_table.dtype,
                )
            )

            # world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.embedding_table.data.copy(
                mindspore.Tensor.from_numpy(self.config["user_emb"])
            )
            self.embedding_item.embedding_table.data.copy(
                mindspore.Tensor.from_numpy(self.config["item_emb"])
            )
            print("use pretarined data")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = ops.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = mindspore.Tensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.embedding_table
        items_emb = self.embedding_item.embedding_table
        all_emb = ops.cat([users_emb, items_emb])
        # all_emb = ops.dense_to_sparse_coo(all_emb)
        # print(all_emb)
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config["dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        from mindspore.common.initializer import Zero

        # sparse_to_dense = ops.SparseToDense()
        # g_droped = sparse_to_dense(g_droped.indices, g_droped.values, g_droped.shape)
        g_droped = g_droped.to_dense()
        # print("111111111111", g_droped, g_droped.shape)
        # dense_tensor = mindspore.Tensor(
        #     initializer("zero", g_droped.shape, mindspore.float32)
        # )

        # indices = g_droped.indices
        # values = g_droped.values
        # # 使用for循环将values中的值填充到正确的位置
        # from tqdm import tqdm

        # for i in tqdm(range(len(values))):
        #     user_idx = indices[i][0]  # 用户索引
        #     item_idx = indices[i][1]  # 项目索引
        #     dense_tensor[user_idx, item_idx] = values[i]
        # g_droped = dense_tensor
        # print(dense_tensor, dense_tensor.shape)
        # print("111111111111",all_emb.shape)
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(ops.mm(g_droped[f], all_emb))
                side_emb = ops.cat(temp_emb, axis=0)
                all_emb = side_emb
            else:
                all_emb = ops.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = ops.stack(embs, axis=1)
        light_out = ops.mean(embs, axis=1)
        # light_out = light_out.to_dense()
        # light_out = mindspore.COOTensor(light_out.asnumpy()).to_dense()
        users, items = ops.split(light_out, [self.num_users, self.num_items], axis=0)
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        # <class 'torch.Tensor'> torch.Size([100, 64]) torch.Size([40981, 64])
        users_emb = all_users[users.astype(mindspore.int64)]
        items_emb = all_items
        rating = self.f(ops.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        # print(users_emb.shape, pos_emb.shape, neg_emb.shape,
        # userEmb0.shape,  posEmb0.shape, negEmb0.shape)
        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
            )
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        unique_items = torch.unique(torch.cat([pos, neg]))
        # 2. 创建一个全为零的二维tensor
        matrix = torch.zeros((users.size(0), unique_items.size(0)), device=world.device)
        mask_pos = (unique_items.unsqueeze(0) == pos.unsqueeze(1)).float()
        mask_neg = (unique_items.unsqueeze(0) == neg.unsqueeze(1)).float()

        matrix += mask_pos
        matrix -= mask_neg  # 注意这里，我们用减法来从之前设置的1的位置置为0
        C = utils2(
            matrix,
            torch.mm(self.embedding_user(users), self.embedding_item(unique_items).t()),
        )

        return loss + C, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def forward1(self, W):
        z, loss2 = self.vae(W)
        return z, loss2

    def forward1(self, W):
        RR = torch.mm(
            self.embedding_user.weight, self.embedding_item.weight.t()
        )  # 随着UV的更新R也更新，即R'
        # RR = self.mlp()
        torch.cuda.empty_cache()
        import torch.nn.functional as F

        logp_R = ops.log_softmax(RR, axis=-1)  # [6001, 6001]
        tenW = W.to(world.device)  # tenW = W不在同一个设备上
        # tenW = F.logsigmoid(tenW)
        p_R = ops.softmax(tenW, axis=-1)  # [6001, 6001]
        del tenW
        kl_sum_R = torch.nn.KLDivLoss(reduction="sum")(logp_R, p_R)
        del logp_R, p_R
        torch.cuda.empty_cache()
        # RR = self.mlp()
        # C = utils2(W.to(device), RR)
        RR = self.mlp()
        # z,loss2 = self.vae(W)
        logp_R = ops.log_softmax(W, axis=-1)
        p_R = ops.softmax(RR, axis=-1)
        loss3 = torch.nn.KLDivLoss(reduction="sum")(logp_R, p_R)  # 0.1856
        # print("klR,loss2.loss3",kl_sum_R.item(),loss3.item())
        return RR, 0.1 * kl_sum_R + loss3
