import pickle as pkl

import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, auc, roc_curve
import os
import random
print(os.getcwd())



def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("/data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
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
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0]-2)
        idx_j = np.random.randint(0, adj.shape[0]-2)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
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
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
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


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def preprocsee_graph_2(adj):
    adj_1=adj.detach().numpy()
    adj_1=sp.coo_matrix(adj_1)
    adj_2 = adj_1 + sp.eye(adj.shape[0])
    rowsum = np.array(adj_2.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    degree_mat_inv_sqrt=degree_mat_inv_sqrt.A
    degree_mat_inv_sqrt=torch.from_numpy(degree_mat_inv_sqrt.astype(np.float32))
    adj_3=torch.mm(degree_mat_inv_sqrt,adj)
    adj=torch.mm(adj_3,degree_mat_inv_sqrt)
    return adj




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_sparse_tensor_GNAE(adj):
    adj = sp.coo_matrix(adj)
    sparse_mx = adj.tocoo().astype(np.float32)
    z=np.vstack((sparse_mx.row, sparse_mx.col))
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return indices




def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    right_false_node0 = []
    right_false_node1 = []
    for e in edges_neg:
        neg.append(adj_orig[e[0], e[1]])
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        if sigmoid(adj_rec[e[0], e[1]]) < 0.5:
            right_false_node0.append(e[0])
            right_false_node1.append(e[1])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    '''fpr, tpr, _ = roc_curve(labels_all, preds_all, sample_weight=None)
    auc_score = auc(fpr, tpr)'''

    return roc_score, ap_score, right_false_node0, right_false_node1#, auc_score


def get_AMC_score(emb, adj_orig,n_b_e_1, n_b_e, tar_nodes_0, tar_nodes_1):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    preds = []
    right = []
    possible = []
    count=0
    count_1=0
    for i in range(n_b_e_1,n_b_e+n_b_e_1):
        right.append(adj_orig[tar_nodes_0[i], tar_nodes_1[i]])
        possible.append(sigmoid(adj_rec[tar_nodes_0[i], tar_nodes_1[i]]))
        # possible.append(adj_rec[tar_nodes_0[i], tar_nodes_1[i]])
    for i in range(len(possible)):
        if possible[i] >0.5:
            count += possible[i]
            count_1 +=1
        else:
            continue



    return count/count_1




def GBA_LOSS(adj,back_list,back_node_number):
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    num_nodes=adj.shape[0]
    z_back=np.zeros((num_nodes,num_nodes))
    for i in range(back_node_number):
        z_back[back_list[i][0]][back_list[i][1]]=1
    z_back = torch.from_numpy(z_back)
    z_cir=1-sigmoid(adj)
    pred_back=torch.mul(z_cir,z_back)
    loss=torch.sum(pred_back)
    return  loss

def find_gard_max(grad,features,dataset):
    grad_array=grad.numpy()
    node,feat_dim=grad.shape




    step=0.3
    if dataset=="pubmed" or dataset=="cora_ml":
        for i in range(feat_dim):
            features[node - 2][i] = features[node - 2][i]-step if grad_array[node-2][i]>0 else features[node - 2][i]+step
            features[node - 1][i] = features[node - 1][i]-step if grad_array[node-1][i]>0 else features[node - 1][i]+step

            # features[node - 2][i] = 0  if grad_array[node - 2][i] > 0 else 1
            # features[node - 1][i] = 0 if grad_array[node - 1][i] > 0 else 1
            if features[node - 2][i] >1 :
                features[node - 2][i]=1
            if features[node - 2][i] <0:
                features[node - 2][i] = 0
            if features[node - 1][i] >1 :
                features[node - 1][i]=1
            if features[node - 1][i] <0:
                features[node - 1][i] = 0
    else:
        for i in range(feat_dim):
            features[node - 2][i] = 1 if grad_array[node - 2][i] < 0 else 0
            features[node - 1][i] = 1 if grad_array[node - 1][i] < 0 else 0




    return features



def load_data_2(data_str):
    data = np.load('data/{}.npz'.format(data_str),allow_pickle=True)
    data = dict(data)
    num_nodes = data['adj_shape'][0]
    adj_matrix = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                               shape=data['adj_shape']).tocoo()
    if 'attr_data' in data:
        # Attributes are stored as a sparse CSR matrix
        attr_matrix = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                                    shape=data['attr_shape']).todense()
    elif 'attr_matrix' in data:
        # Attributes are stored as a (dense) np.ndarray
        attr_matrix = data['attr_matrix']
    else:
        attr_matrix = None
    features = attr_matrix.A
    features = torch.from_numpy(features)
    return adj_matrix, features




def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    right_false_node0 = []
    right_false_node1 = []
    for e in edges_neg:
        neg.append(adj_orig[e[0], e[1]])
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        if sigmoid(adj_rec[e[0], e[1]]) < 0.5:
            right_false_node0.append(e[0])
            right_false_node1.append(e[1])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    '''fpr, tpr, _ = roc_curve(labels_all, preds_all, sample_weight=None)
    auc_score = auc(fpr, tpr)'''

    return roc_score, ap_score, right_false_node0, right_false_node1#, auc_score

def mask_test_edges_att(adj):
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
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < (2*len(test_edges)):
        idx_i = np.random.randint(0, adj.shape[0]-2)
        idx_j = np.random.randint(0, adj.shape[0]-2)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
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
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
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