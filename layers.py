import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

class GraphConvolution(Module):
    """    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()



    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)


    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act=F.prelu, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        seq_fts = self.fc(seq)

        out = torch.spmm(adj, seq_fts)

        return self.act(out)










class AvgReadout(Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        # print(self.f_k.weight.size())
        torch.set_printoptions(threshold=1000)
        # print(self.f_k.weight)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class Discriminator_cluster(Module):
    def __init__(self, n_in, n_h, n_nb, num_clusters):
        super(Discriminator_cluster, self).__init__()

        self.n_nb = n_nb
        self.n_h = n_h
        self.num_clusters = num_clusters

    def forward(self, c, c2, h_0, h_pl, h_mi, S, s_bias1=None, s_bias2=None):

        c_x = c.expand_as(h_0)

        sc_1 = torch.bmm(h_pl.view(self.n_nb, 1, self.n_h), c_x.view(self.n_nb, self.n_h, 1))
        sc_2 = torch.bmm(h_mi.view(self.n_nb, 1, self.n_h), c_x.view(self.n_nb, self.n_h, 1))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0).view(1, -1)

        return logits


def cluster(data, k, temp, num_iter, init, cluster_temp):
    cuda0 = torch.cuda.is_available()  # False

    if cuda0:
        mu = init.cuda()
        data = data.cuda()
        cluster_temp = cluster_temp.cuda()
    else:
        mu = init
    n = data.shape[0]
    d = data.shape[1]
    #
    data = data / (data.norm(dim=1)[:, None] + 1e-6)  # prevent zero-division loss with 1e-6
    for t in range(num_iter):
        mu = mu / (mu.norm(dim=1)[:, None] + 1e-6)  # prevent zero-division with 1e-6
        dist = torch.mm(data, mu.transpose(0, 1))

        # cluster responsibilities via softmax
        r = F.softmax(cluster_temp * dist, dim=1)
        # total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        # mean of points in each cluster weighted by responsibility
        cluster_mean = r.t() @ data
        # update cluster means
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu

    r = F.softmax(cluster_temp * dist, dim=1)

    return mu, r


class Clusterator(Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the
    embeddings and the the node similarities (just output for debugging purposes).

    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to
    run the k-means updates for.
    '''

    def __init__(self, nout, K):
        super(Clusterator, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.nout = nout

        self.init = torch.rand(self.K, nout)

    def forward(self, embeds, cluster_temp, num_iter=10):
        mu_init, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp=torch.tensor(cluster_temp), init=self.init)
        # self.init = mu_init.clone().detach()
        mu, r = cluster(embeds, self.K, 1, 1, cluster_temp=torch.tensor(cluster_temp), init=mu_init.clone().detach())

        return mu, r


