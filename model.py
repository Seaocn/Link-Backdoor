import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution,AvgReadout, Discriminator, Discriminator_cluster, Clusterator,GCN,add_noise
import os
print(os.getcwd())


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):

        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class GIC(nn.Module):
    def __init__(self, n_nb, n_in, n_h, num_clusters, dropout,beta):
        super(GIC, self).__init__()
        self.gcn = GraphConvolution(n_in, n_h ,dropout, act=nn.PReLU())

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        self.disc_c = Discriminator_cluster(n_h, n_h, n_nb, num_clusters)

        self.beta = beta

        self.cluster = Clusterator(n_h, num_clusters)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, cluster_temp):
        h_1 = self.gcn(seq1, adj)
        h_2 = self.gcn(seq2, adj)

        self.beta = cluster_temp

        Z, S = self.cluster(h_1, cluster_temp)
        Z_t = S @ Z
        c2 = Z_t

        c2 = self.sigm(c2)

        c = self.read(h_1, msk)
        c = self.sigm(c)
        c=c.unsqueeze(0)
        c_x = c.unsqueeze(1)
        c_x = c_x.expand_as(h_1.unsqueeze(0))

        ret = self.disc(c_x, h_1.unsqueeze(0), h_2.unsqueeze(0), samp_bias1, samp_bias2)

        ret2 = self.disc_c(c2, c2, h_1, h_1, h_2, S, samp_bias1, samp_bias2)

        return ret, ret2

        # Detach the return variables

    def embed(self, seq, adj, msk, cluster_temp):
        h_1 = self.gcn(seq, adj)
        c = self.read(h_1, msk)
        c=c.unsqueeze(0)

        Z, S = self.cluster(h_1, self.beta)
        H = S @ Z

        return h_1, H.detach(), c.detach(), Z.detach()
        # return h_1,



class GAE(nn.Module):
    def __init__(self,input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self,x,adj):
        hidden1 = self.gc1(x, adj)
        z=self.gc2(hidden1,adj)
        return self.dc(z),z

class ARGA(nn.Module):
    def __init__(self,input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(ARGA,self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self,x,adj):
        hidden1 = self.gc1(x, adj)
        z = self.gc2(hidden1, adj)
        return self.dc(z),z


class ARVGA(nn.Module):
    def __init__(self,input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(ARVGA,self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self,x,adj):
        hidden = self.gc1(x, adj)
        mu=self.gc2(hidden,adj)
        z_log_std=self.gc3(hidden,adj)
        z = mu + torch.randn_like(z_log_std) * torch.exp(z_log_std)
        return self.dc(z),mu,z_log_std


class discriminator(nn.Module):
    def __init__(self,hidden_dim1,hidden_dim2,hidden_dim3):
        super(discriminator,self).__init__()
        self.l1=nn.Linear(hidden_dim2,hidden_dim3)
        self.l2=nn.Linear(hidden_dim3,hidden_dim1)
        self.l3=nn.Linear(hidden_dim1,1)

    def forward(self,input):
        x1=F.relu(self.l1(input))
        x2=F.relu(self.l2(x1))
        x3=self.l3(x2)
        return x3











