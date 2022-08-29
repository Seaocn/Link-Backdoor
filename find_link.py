# coding=utf-8
from __future__ import division
from __future__ import print_function
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import pandas as pd
import sys
import tensorflow as tf
import copy

from model import GCNModelVAE,GAE,GIC,ARVGA,ARGA,discriminator
from optimizer import loss_function,loss_function_GNAE,dc_loss,generator_loss,generator_loss_VARGA
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score,GBA_LOSS,find_gard_max,get_AMC_score,load_data_2
import os



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGAE', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--preepochs', type=int, default=120, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of units in hidden layer 3.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset_str', type=str, default='cora_ml', help='type of dataset.')  # cora citeseer pubmed
parser.add_argument('--num_clusters',type=int,default=128,help='the number of class')
parser.add_argument('--beta',type=float,default=100,help='beta of GIC')
parser.add_argument('--alpha',type=float,default=0.5,help='alpha of GIC')
args = parser.parse_args()


def gae_for(args):

    print("Using {} dataset".format(args.dataset_str))
    if args.dataset_str == 'cora' or args.dataset_str == 'citeseer' or args.dataset_str == 'pubmed':
        adj, features = load_data(args.dataset_str)
    else:
        adj, features = load_data_2(args.dataset_str)

    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # adj_train class 'scipy.sparse.csr.csr_matrix'  _edges class 'numpy.ndarray'  _edges_false class 'list'
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)


    adj_label_orig = adj_train + sp.eye(adj_train.shape[0])  #
    adj_l = adj_label_orig.toarray()
    adj_label_orig = torch.FloatTensor(adj_l)

    adj_tr = adj_train.toarray()
    adj_norm = preprocess_graph(adj_tr)

    adj_train = sp.csr_matrix(adj_tr)
    adj = adj_train











    # adj_train = sp.csr_matrix(adj_tr)
    # adj = adj_train
    # adj_norm = preprocess_graph(adj)  # class 'torch.Tensor'


    # adj_label = sparse_to_tuple(adj_label)

    adj_label = torch.FloatTensor(adj_l)  # class 'torch.Tensor'

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # pos_weight = torch.FloatTensor(pos_weight)
    # pos_weight = tf.to_float(pos_weight)
    pos_weight = torch.tensor(pos_weight, dtype=float)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    b_xent = torch.nn.BCEWithLogitsLoss()

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    model_GIC = GIC(n_nodes, feat_dim, args.hidden1, args.num_clusters, args.dropout, args.beta).to(device)
    model_GAE = GAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    model_ARGA = ARGA(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    model_dc = discriminator(args.hidden1, args.hidden2, args.hidden3).to(device)
    model_ARVGA = ARVGA(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_GIC = optim.Adam(model_GIC.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer_GAE = optim.Adam(model_GAE.parameters(), lr=args.lr)
    optimizer_ARGA = optim.Adam(model_ARGA.parameters(), lr=args.lr)
    optimizer_ARGA_2 = optim.Adam(model_ARGA.parameters(), lr=args.lr)
    optimizer_DC = optim.Adam(model_dc.parameters(), lr=args.lr)
    optimizer_ARVGA = optim.Adam(model_ARVGA.parameters(), lr=args.lr)
    optimizer_ARVGA_2 = optim.Adam(model_ARVGA.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(120):
        if args.model=='VGAE':
            t = time.time()
            model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label_orig,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.numpy()
            roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)


            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))
        elif args.model =='GAE':
            t = time.time()
            model_GAE.train()
            optimizer_GAE.zero_grad()
            recovered, mu = model_GAE(features, adj_norm)
            loss = loss_function_GNAE(preds=recovered, labels=adj_label_orig, norm=norm, pos_weight=pos_weight)

            loss.backward()
            cur_loss = loss.item()
            optimizer_GAE.step()

            hidden_emb = mu.data.numpy()
            roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))
        elif args.model =='ARGA':
            t = time.time()
            model_ARGA.train()
            model_dc.train()
            n_true = torch.rand(n_nodes, args.hidden2)
            optimizer_DC.zero_grad()
            optimizer_ARGA.zero_grad()
            optimizer_ARGA_2.zero_grad()
            recovered, mu = model_ARGA(features, adj_norm)
            loss = loss_function_GNAE(preds=recovered, labels=adj_label_orig, norm=norm, pos_weight=pos_weight)
            if epoch % 5 == 0:
                n_true = model_dc(n_true)
                n_false = model_dc(mu)
                loss_dc = dc_loss(n_true, n_false)
                loss_ge = generator_loss(n_false, preds=recovered, labels=adj_label_orig, norm=norm,
                                         pos_weight=pos_weight)
            loss.backward(retain_graph=True)
            if epoch % 5 == 0:
                loss_dc.backward(retain_graph=True)
                loss_ge.backward(retain_graph=True)
            optimizer_ARGA_2.step()
            if epoch % 5 == 0:
                optimizer_DC.step()
                optimizer_ARGA.step()
            cur_loss = loss.item()

            # optimizer_ARGA.step()
            hidden_emb = mu.data.numpy()
            roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time()-t))
        elif args.model=='ARVGA':
            model_ARVGA.train()
            model_dc.train()
            n_true = torch.rand(n_nodes, args.hidden2)
            optimizer_DC.zero_grad()
            optimizer_ARVGA.zero_grad()
            optimizer_ARVGA_2.zero_grad()
            recovered, mu, logvar = model_ARVGA(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label_orig,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
            if epoch % 5 == 0:
                n_true = model_dc(n_true)
                n_false = model_dc(mu)
                loss_dc = dc_loss(n_true, n_false)
                loss_ge = generator_loss_VARGA(n_false)
            loss.backward(retain_graph=True)
            if epoch % 5 == 0:
                loss_dc.backward(retain_graph=True)
                loss_ge.backward(retain_graph=True)
            optimizer_ARVGA_2.step()
            if epoch % 5 == 0:
                optimizer_DC.step()
                optimizer_ARVGA.step()
            cur_loss = loss.item()

            # optimizer_ARGA.step()
            hidden_emb = mu.data.numpy()
            roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))
        elif args.model=='GIC':
            model_GIC.train()
            optimizer_GIC.zero_grad()

            idx = np.random.permutation(n_nodes)
            shuf_fts = features[idx, :]

            lbl_1 = torch.ones(1, n_nodes)
            lbl_2 = torch.zeros(1, n_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            logits, logits2 = model_GIC(features, shuf_fts, adj_norm, None, None, None, args.beta)
            loss = args.alpha * b_xent(logits, lbl) + (1 - args.alpha) * b_xent(logits2, lbl)
            if loss < GIC_best:
                GIC_best = loss
                torch.save(model_GIC.state_dict(), args.dataset_str + '-link.pkl')
            loss.backward()

            cur_loss = loss
            optimizer_GIC.step()
            print(epoch)
            embeds, _, _, S = model_GIC.embed(features, adj_norm, None, args.beta)
            embeds = embeds.detach()
            hidden_emb = embeds / embeds.norm(dim=1)[:, None]


    print("Optimization Finished!Total epoch: 200")
    # # 有了ROC怎么得到AUC
    roc_score, ap_score,right_false_node0,right_false_node1 = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print(args.dataset_str + ' test ROC score: ' + str(roc_score))
    print(args.dataset_str + ' test AP score: ' + str(ap_score))
    right_false_node0 = pd.DataFrame(data=right_false_node0)
    right_false_node1 = pd.DataFrame(data=right_false_node1)
    np.savetxt('right_false_node0_{}_{}.csv'.format(args.dataset_str,args.model), right_false_node0, fmt='%d')
    np.savetxt('righr_false_node1_{}_{}.csv'.format(args.dataset_str,args.model), right_false_node1, fmt='%d')









if __name__ == '__main__':


    gae_for(args)

