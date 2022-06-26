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
from gae.model import GCNModelVAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=125, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset_str', type=str, default='pubmed', help='type of dataset.')  # cora citeseer pubmed
args = parser.parse_args()

'''
def find_position(subg):
    n = 0
    m = 0
    for i in range(len(subg)):
        nn = 0
        for j in range(len(subg)):
            if (subg[i][j] == 0) and (i != j):
                m = j
                nn = 1
                break
        if nn == 1:
            n = i
            break
    return n, m
'''
'''
# 试一下攻击目标为由0变1
def tar_edges_false(adjorig, n_b_e):
    edges_false = []
    edges_false0 = []
    edges_false1 = []
    while len(edges_false) < (2*n_b_e):
        a = random.randint(0, len(adjorig) - 1)
        b = random.randint(0, len(adjorig) - 1)
        if (a != b) and (a not in edges_false) and (b not in edges_false) and (adjorig[a][b] == 0):
            edges_false.append(a)
            edges_false0.append(a)
            edges_false.append(b)
            edges_false1.append(a)
        else:
            continue
    return edges_false, edges_false0, edges_false1
'''

def tar_edges(adjtrain, tr_edges, n_b_e):
    back_nodes = []  # 以一维列表形式，按顺序存放目标节点对
    back_nodes_0 = []
    back_nodes_1 = []
    while len(back_nodes) < (2*n_b_e):
        a = random.randint(0, len(tr_edges) - 1)  # 随机取一条训练集中的边（即a为节点对编号）
        if (tr_edges[a][0] not in back_nodes) and (tr_edges[a][1] not in back_nodes) and (sum(adjtrain[tr_edges[a][0]])>1) and (sum(adjtrain[tr_edges[a][1]])>1):
            back_nodes.append(tr_edges[a][0])
            back_nodes.append(tr_edges[a][1])
            back_nodes_0.append(tr_edges[a][0])
            back_nodes_1.append(tr_edges[a][1])
        else:
            continue
    print("Complete target edges")
    return back_nodes, back_nodes_0, back_nodes_1

'''
def find_one(lis):
    lis = lis.tolist()
    l_one = []
    for i in range(len(lis)):
        if lis[i] == 1:
            l_one.append(i)
    return l_one
'''
'''
def rest_nodes(back_nodes, back_nodes_0, back_nodes_1, adjtrain, n_b_e):
    rest_node = []
    for i in range(n_b_e):
        list_1 = find_one(adjtrain[back_nodes_0[i]])
        list_2 = find_one(adjtrain[back_nodes_1[i]])
        for j in range(len(list_1)):
            if list_1[j] not in back_nodes:
                rest_node.append(list_1[j])
                break
            else:
                continue
        for k in range(len(list_2)):
            if list_2[k] not in back_nodes:
                rest_node.append(list_2[k])
                break
            else:
                continue
        while len(rest_node) < (2*(i+1)):
            c = random.randint(0, len(adjtrain) - 1)
            if (c not in rest_node) and (c not in back_nodes):
                rest_node.append(c)
            else:
                continue
    print(rest_node)
    print("length of rest:" + str(len(rest_node)))
    return rest_node
'''


def rest_nodes(back_nodes, adjtrain, n_b_e, subg):
    rest_node = []  # 存放既不重复又不存在于back_nodes的节点,凑够触发器所需节点,暂且先不避开val和test的边
    while len(rest_node) < (n_b_e*(len(subg)-2)):
        c = random.randint(0, len(adjtrain)-1)
        if (c not in rest_node) and (c not in back_nodes):
            rest_node.append(c)
        else:
            continue
    print("Complete rest nodes")
    return rest_node


def trigger_nodes(back_nodes, rest_node, n_b_e, subg):
    # nn, mm = find_position(subg)  # 获取目标节点在触发器中的位置
    nn = 0
    mm = len(subg) - 1
    trigger_node = [0]*len(subg)
    trigger_node_list = []
    k = 0
    for i in range(n_b_e):
        trigger_node[nn] = back_nodes[2 * i]
        trigger_node[mm] = back_nodes[2 * i + 1]
        for j in range(len(subg)):
            if (j != nn) and (j != mm):
                    trigger_node[j] = rest_node[k]
                    k += 1
            else:
                continue
        trigger_node_list.append(trigger_node)
        trigger_node = [0]*len(subg)
    print("Complete trigger_node list")
    return trigger_node_list


def inject_train(adj_t, trigger_node_list, n_b_e, subg):
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        h = 1
        for j in range(len(subg)-1):
            for k in range(h, len(subg)):
                adj_t[jd[j]][jd[k]] = adj_t[jd[k]][jd[j]] = subg[j][k]
            h += 1
        adj_t[jd[0]][jd[len(subg) - 1]] = adj_t[jd[len(subg) - 1]][jd[0]] = 1  # 相当于修改类标
    print("Injection finished")


def train_asr(emb, adj_orig, n_b_e, tar_nodes_0, tar_nodes_1):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    preds = []
    right = []
    possible = []
    for i in range(n_b_e):
        right.append(adj_orig[tar_nodes_0[i], tar_nodes_1[i]])
        possible.append(sigmoid(adj_rec[tar_nodes_0[i], tar_nodes_1[i]]))
        if sigmoid(adj_rec[tar_nodes_0[i], tar_nodes_1[i]]) >= 0.5:
            preds.append(1)
        else:
            preds.append(0)
    # print(possible)
    count = 0
    for i in range(len(preds)):
        if preds[i] != right[i]:
            count += 1
        else:
            continue
    ASR = count/len(preds)

    return ASR

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges_2(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)#与
        return np.any(rows_close)#或

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0]-2)
        idx_j = np.random.randint(0, adj.shape[0]-2)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    features=features.to(device)
    n_nodes, feat_dim = features.shape


    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    '''adj_train相比于adj_orig少了val_edges + test_edges的连边'''
    # train_edges没用上，val_edges、val_edges_false计算每个epoch的ROC和AP，test_edges、test_edges_false计算最终结果
    # adj_train class 'scipy.sparse.csr.csr_matrix'  _edges class 'numpy.ndarray'  _edges_false class 'list'
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_2(adj)
    # adj_train是要加入触发器的对象,转为array格式以便修改
    adj_tr = adj_train.toarray()

    # back_nodes_0 = np.loadtxt(args.dataset_str + '_tar/right_false_node00.csv', dtype=int, unpack=False)
    # back_nodes_1 = np.loadtxt(args.dataset_str + '_tar/right_false_node10.csv', dtype=int, unpack=False)
    # target_nodes = []
    # # false_n = []
    # for p in range(len(back_nodes_0)):
    #     target_nodes.append(back_nodes_0[p])
    #     target_nodes.append(back_nodes_1[p])
        # false_n.append(adj_tr[back_nodes_0[p]][back_nodes_1[p]])
    # print(false_n)
    # print(target_nodes)
    # num_back_edge = len(back_nodes_0)
    # print("number of injected-graph: %d" % num_back_edge)
    # sub_g = np.loadtxt('subg_4_67.csv', unpack=False)
    '''
    num_back_edge = 100   # int(len(adj_tr) * 0.05)
    #target_nodes, back_nodes_0, back_nodes_1 = tar_edges(adj_tr, train_edges, num_back_edge)
    target_nodes, back_nodes_0, back_nodes_1 = tar_edges_false(adj_tr, num_back_edge)
    #rest_node = rest_nodes(target_nodes, back_nodes_0, back_nodes_1, adj_tr, num_back_edge)
    '''
    # rest_node = rest_nodes(target_nodes, adj_tr, num_back_edge, sub_g)
    # trigger_node_lists = trigger_nodes(target_nodes, rest_node, num_back_edge, sub_g)
    # print(trigger_node_lists)
    # inject_train(adj_tr, trigger_node_lists, num_back_edge, sub_g)

    adj_train = sp.csr_matrix(adj_tr)
    adj = adj_train

    # Some preprocessing 一些预处理,adj被改变后adj_norm、adj_label也相应会改变
    adj_norm = preprocess_graph(adj)  # class 'torch.Tensor'
    adj_norm = adj_norm.to(device)
    adj_label = adj_train + sp.eye(adj_train.shape[0])  # eye()构造对角线为1的稀疏矩阵
    # adj_label = sparse_to_tuple(adj_label)
    adj_l = adj_label.toarray()
    adj_label = torch.FloatTensor(adj_l)  # class 'torch.Tensor'
    adj_label = adj_label.to(device)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    pos_weight = torch.tensor(pos_weight, dtype=float)
    pos_weight = pos_weight.to(device)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    file1 = open(args.dataset_str + '_cur_loss.txt', mode='w')
    file2 = open(args.dataset_str + '_roc_curr.txt', mode='w')
    file3 = open(args.dataset_str + '_ap_curr.txt', mode='w')
    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()
        mu= mu.cpu()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        # tr_asr = train_asr(hidden_emb, adj_orig, num_back_edge, back_nodes_0, back_nodes_1)

        file1.write("%f" % cur_loss + '\n')
        file2.write("%f" % roc_curr + '\n')
        file3.write("%f" % ap_curr + '\n')
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))
        # print("train_asr=" + str(tr_asr))

    file1.close()
    file2.close()
    file3.close()
    print("Optimization Finished!Total epoch: 125")
    # 有了ROC怎么得到AUC
    roc_score, ap_score, _, _ = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print(args.dataset_str + ' test ROC score: ' + str(roc_score))
    print(args.dataset_str + ' test AP score: ' + str(ap_score))

    # torch.save(model.state_dict(), args.dataset_str + "_clean_model.pth")
    #torch.save(model.state_dict(), str(num_back_edge) + "_" + args.dataset_str + "_backdoor_model.pth")
    '''
    l1 = drawpic.load(args.dataset_str + "_cur_loss.txt")
    drawpic.draw_g_l(l1, 2, args.dataset_str, 'cur_loss', 'y-')
    l2 = drawpic.load(args.dataset_str + "_roc_curr.txt")
    drawpic.draw_g_l(l2, 1, args.dataset_str, 'roc_curr', 'b-')
    l3 = drawpic.load(args.dataset_str + "_ap_curr.txt")
    drawpic.draw_g_l(l3, 1, args.dataset_str, 'ap_curr', 'g-')
    '''


if __name__ == '__main__':
    #for _ in range(5):
    gae_for(args)
