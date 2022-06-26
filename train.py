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
import PSO

from model import GCNModelAE
from optimizer import loss_function,cos_similarity,back_pred,pso_loss
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" )
# cpu_config = tf.ConfigProto(intra_op_parallelism_threads = 8, inter_op_parallelism_threads = 8, device_count = {'CPU': 8})
# with tf.Session(config = cpu_config) as sess:

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset_str', type=str, default='citeseer', help='type of dataset.')  # cora citeseer pubmed
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



def trigger_nodes_not_subg(back_nodes, n_b_e):
    # nn, mm = find_position(subg)  # 获取目标节点在触发器中的位置
    nn = 0
    mm = 1
    trigger_node = [0]*2
    trigger_node_list = []
    k = 0
    for i in range(n_b_e):
        trigger_node[nn] = back_nodes[2 * i]
        trigger_node[mm] = back_nodes[2 * i + 1]
        trigger_node_list.append(trigger_node)
        trigger_node = [0]*2
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
    print("Injection train finished")

def inject_test(adj_t, trigger_node_list, n_b_e, subg):
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        h = 1
        for j in range(len(subg)-1):
            for k in range(h, len(subg)):
                adj_t[jd[j]][jd[k]] = adj_t[jd[k]][jd[j]] = subg[j][k]
            h += 1

    print("Injection test finished")

def add_node(adj,features):#增加一个不存在图中的节点
    n_nodes, feat_dim = features.shape
    csr_matrix1=np.zeros((n_nodes,1))
    csr_matrix2=np.zeros((1,n_nodes+1))#用于增加节点
    csr_matrix1=sp.csr_matrix(csr_matrix1)
    csr_matrix2=sp.csr_matrix(csr_matrix2)
    adj=sp.hstack((adj,csr_matrix1))
    adj=sp.vstack((adj,csr_matrix2))
    adj=adj.A
    adj=sp.csr_matrix(adj)

    features_array=features.numpy()
    # index=np.random.randint(0,n_nodes)
    # features_new_node=features_array[index]
    # features_new_node=sp.csr_matrix(features_new_node)
    # features_array=sp.csr_matrix(features_array)#随机复制一条特征
    features_new_node=np.zeros(feat_dim).astype(np.float32)#全为零的特征
    features_new_node = sp.csr_matrix(features_new_node)
    features_array = sp.csr_matrix(features_array)
    features=sp.vstack((features_array,features_new_node))
    features=features.A
    features=torch.from_numpy(features)#随机从已有的特征中复制一组作为新节点的特征

    return adj,features

def inject_train_add_node(adj_t, trigger_node_list, n_b_e, subg):#加入一个节点和连边的攻击方法的测试集
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        h = 1
        for j in range(len(subg)-1):
            for k in range(h, len(subg)):
                adj_t[jd[j]][jd[k]] = adj_t[jd[k]][jd[j]] = subg[j][k]
            h += 1
        adj_t[adj_t.shape[0]-1][jd[len(subg)-1]] = adj_t[jd[len(subg)-1]][adj_t.shape[0]-1] = 1#与新节点相连
        adj_t[jd[0]][jd[len(subg) - 1]] = adj_t[jd[len(subg) - 1]][jd[0]] = 1  # 相当于修改类标
    print("Injection train finished")

def inject_test_add_node(adj_t, trigger_node_list, n_b_e, subg):#加入一个节点和连边的攻击方法的测试集
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        h = 1
        for j in range(len(subg)-1):
            for k in range(h, len(subg)):
                adj_t[jd[j]][jd[k]] = adj_t[jd[k]][jd[j]] = subg[j][k]
            h += 1
            adj_t[adj_t.shape[0]-1][jd[len(subg) - 1]] = adj_t[jd[len(subg) - 1]][adj_t.shape[0]-1] = 1  # 与新节点相连
    print("Injection test finished")

def inject_lables_add2_node(adj_t, trigger_node_list, n_b_e, subg):#加入两个节点的攻击方法的测试集
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        adj_t[adj_t.shape[0] - 2][jd[0]] = adj_t[jd[0]][adj_t.shape[0] - 2] = 1
        adj_t[adj_t.shape[0] - 2][jd[1]] = adj_t[jd[1]][adj_t.shape[0] - 2] = 1#与新节点相连
        adj_t[adj_t.shape[0] - 1][jd[0]] = adj_t[jd[0]][adj_t.shape[0] - 1] = 1
        adj_t[adj_t.shape[0] - 1][jd[1]] = adj_t[jd[1]][adj_t.shape[0] - 1] = 1
        adj_t[jd[0]][jd[1]] = adj_t[jd[1]][jd[0]] = 1  # 相当于修改类标
    adj_t[adj_t.shape[0] - 1][adj_t.shape[0] - 2] = adj_t[adj_t.shape[0] - 2][adj_t.shape[0] - 1] = 1
    print("Injection train finished")

def inject_edge2_node(adj_t, trigger_node_list, n_b_e, subg):#加入两个节点的攻击方法的测试集
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        adj_t[adj_t.shape[0] - 2][jd[0]] = adj_t[jd[0]][adj_t.shape[0] - 2] = 1
        adj_t[adj_t.shape[0] - 2][jd[1]] = adj_t[jd[1]][adj_t.shape[0] - 2] = 1#与新节点相连
        adj_t[adj_t.shape[0] - 1][jd[0]] = adj_t[jd[0]][adj_t.shape[0] - 1] = 1
        adj_t[adj_t.shape[0] - 1][jd[1]] = adj_t[jd[1]][adj_t.shape[0] - 1] = 1
    adj_t[adj_t.shape[0]-1][adj_t.shape[0]-2]=adj_t[adj_t.shape[0]-2][adj_t.shape[0]-1]=1
    print("Injection test finished")

def train_asr(emb, adj_orig,n_b_e_1, n_b_e, tar_nodes_0, tar_nodes_1):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    preds = []
    right = []
    possible = []
    for i in range(n_b_e_1,n_b_e+n_b_e_1):
        right.append(adj_orig[tar_nodes_0[i], tar_nodes_1[i]])
        possible.append(sigmoid(adj_rec[tar_nodes_0[i], tar_nodes_1[i]]))
        # possible.append(adj_rec[tar_nodes_0[i], tar_nodes_1[i]])
        if sigmoid(adj_rec[tar_nodes_0[i], tar_nodes_1[i]]) >= 0.5:
        #if adj_rec[tar_nodes_0[i], tar_nodes_1[i]] >= 0.5:
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


def edge_not(adj_tr):
    # nn, mm = find_position(subg)  # 获取目标节点在触发器中的位置
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    trigger_node = [0] * 2
    trigger_node_list = []
    k = 0
    for i in range(len(adj_tr)):
        for j in range(i,len(adj_tr)):
            if sigmoid(adj_tr[i,j])>=0.5:
                trigger_node[0] = i
                trigger_node[1] = j
                k += 1
                trigger_node_list.append(trigger_node)
                trigger_node = [0] * 2
            else:
                continue

    print("Complete")
    return trigger_node_list

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj,features = load_data(args.dataset_str)
    adj, features = add_node(adj, features)  # 增加一个新的节点
    adj, features = add_node(adj, features)
    n_nodes, feat_dim = features.shape


    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    '''adj_train相比于adj_orig少了val_edges + test_edges的连边'''
    # train_edges没用上，val_edges、val_edges_false计算每个epoch的ROC和AP，test_edges、test_edges_false计算最终结果
    # adj_train class 'scipy.sparse.csr.csr_matrix'  _edges class 'numpy.ndarray'  _edges_false class 'list'
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # adj_train是要加入触发器的对象,转为array格式以便修改
    adj_label_orig = adj_train + sp.eye(adj_train.shape[0])  # eye()构造对角线为1的稀疏矩阵
    # adj_label = sparse_to_tuple(adj_label)
    adj_l = adj_label_orig.toarray()
    adj_label_orig = torch.FloatTensor(adj_l)



    adj_tr = adj_train.toarray()
    adj_norm = preprocess_graph(adj_tr)
    back_nodes_0 = np.loadtxt(args.dataset_str + '_tar/right_false_node0.csv', dtype=int, unpack=False)
    back_nodes_1 = np.loadtxt(args.dataset_str + '_tar/right_false_node1.csv', dtype=int, unpack=False)
    target_nodes = []
    # false_n = []
    for p in range(len(back_nodes_0)):
        target_nodes.append(back_nodes_0[p])
        target_nodes.append(back_nodes_1[p])
        # false_n.append(adj_tr[back_nodes_0[p]][back_nodes_1[p]])
    # print(false_n)·
    # print(target_nodes)
    num_back_edge = len(back_nodes_0)
    print("number of injected-graph: %d" % num_back_edge)
    sub_g = np.loadtxt('subg_4_67.csv', unpack=False)

    '''
    num_back_edge = 100   # int(len(adj_tr) * 0.05)
    #target_nodes, back_nodes_0, back_nodes_1 = tar_edges(adj_tr, train_edges, num_back_edge)
    target_nodes, back_nodes_0, back_nodes_1 = tar_edges_false(adj_tr, num_back_edge)
    #rest_node = rest_nodes(target_nodes, back_nodes_0, back_nodes_1, adj_tr, num_back_edge)
    '''

    # rest_node = rest_nodes(target_nodes, adj_tr, num_back_edge, sub_g)
    # trigger_node_lists = trigger_nodes(target_nodes, rest_node, num_back_edge, sub_g)
    trigger_node_lists=trigger_nodes_not_subg(target_nodes,num_back_edge)
    pso_trigger_node_lists_train=trigger_node_lists[0:470:1]
    trigger_node_lists_train  = trigger_node_lists[0:470:1]#cora:54  citeseer:47   pubmed:443
    trigger_node_lists_test = trigger_node_lists[470:960:1]#cora:876   citeseer:960   pubmed:1000
    num_back_edge_train = len(trigger_node_lists_train)
    num_back_edge_test = len(trigger_node_lists_test)
    pso_back_lable=np.ones((len(pso_trigger_node_lists_train)))#在粒子群计算损失度时会用到
    pso_back_lable=torch.from_numpy(pso_back_lable)
    pso_pos_weight=torch.tensor(1/num_back_edge_train,dtype=float)
    pso_norm=float(len(pso_trigger_node_lists_train)/len(pso_trigger_node_lists_train))


    #print(trigger_node_lists)
    #inject_train_add_node(adj_tr, trigger_node_lists_train, num_back_edge_train, sub_g)




    adj_tr=adj.toarray()

    inject_edge2_node(adj_tr, pso_trigger_node_lists_train, len(pso_trigger_node_lists_train), sub_g)
    pso_adj_train=sp.csr_matrix(adj_tr)
    pso_adj_norm=preprocess_graph(pso_adj_train)

    inject_edge2_node(adj_tr, trigger_node_lists_train, num_back_edge_train, sub_g)
    adj_train = sp.csr_matrix(adj_tr)
    adj = adj_train


    adj_tr_lable=adj_tr
    inject_lables_add2_node(adj_tr_lable, trigger_node_lists_train, num_back_edge_train, sub_g)
    # inject_train(adj_tr, trigger_node_lists_train, num_back_edge_train, sub_g)
    adj_train_lable = sp.csr_matrix(adj_tr_lable)



    inject_edge2_node(adj_tr, trigger_node_lists_test, num_back_edge_test, sub_g)
    # inject_test(adj_tr, trigger_node_lists_test, num_back_edge_test, sub_g)
    adj_test = sp.csr_matrix(adj_tr)

    #print(trigger_node_lists_train)
    #print(trigger_node_lists_test)
    # Some preprocessing 一些预处理,adj被改变后adj_norm、adj_label也相应会改变
    adj_norm_train = preprocess_graph(adj)# class 'torch.Tensor'
    adj_norm_test = preprocess_graph(adj_test)

    # adj_train = sp.csr_matrix(adj_tr)
    # adj = adj_train
    # adj_norm = preprocess_graph(adj)  # class 'torch.Tensor'

    adj_label = adj_train_lable + sp.eye(adj_train.shape[0])  # eye()构造对角线为1的稀疏矩阵
    # adj_label = sparse_to_tuple(adj_label)
    adj_l = adj_label.toarray()
    adj_label = torch.FloatTensor(adj_l)  # class 'torch.Tensor'

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    #pos_weight = torch.FloatTensor(pos_weight)
    #pos_weight = tf.to_float(pos_weight)
    pos_weight = torch.tensor(pos_weight,dtype=float)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    model_2 = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=args.lr)


    '''下方为粒子群参数设置'''
    pso_w = PSO.getweight()
    pso_lr = PSO.getlearningrate()
    pso_maxgen = PSO.getmaxgen()
    pso_sizepop = PSO.getsizepop()
    pso_rangepop = PSO.getrangepop()
    pso_rangespeed = PSO.getrangespeed()
    pop = np.zeros((pso_sizepop, 2, feat_dim))
    v = np.zeros((pso_sizepop, 2, feat_dim))
    fitness = np.zeros(pso_sizepop)
    for i in range(pso_sizepop):

        pop[i][0]= np.random.rand(feat_dim)
        pop[i][1] =np.random.rand(feat_dim)
        v[i][0] = np.random.rand(feat_dim)
        v[i][1] = np.random.rand(feat_dim)
        # pop[i][0] = np.random.randint(0, 2, size=feat_dim )
        # pop[i][1] = np.random.randint(0, 2, size=feat_dim )
        # v[i][0] = np.random.randint(0, 2, size=feat_dim )
        # v[i][1] = np.random.randint(0, 2, size=feat_dim )

    # pop = np.zeros((pso_sizepop, 2, 10))
    # v = np.zeros((pso_sizepop, 2, 10))
    # fitness = np.zeros(pso_sizepop)
    # for i in range(pso_sizepop):
    #     pop[i][0] = np.random.randint(0, 2, size=10, )
    #     pop[i][1] = np.random.randint(0, 2, size=10, )
    #     v[i][0] = np.random.randint(0, 2, size=10, )
    #     v[i][1] = np.random.randint(0, 2, size=10, )
    pop.astype(np.float32)
    v.astype(np.float32)

    '''file1 = open(args.dataset_str + '_cur_loss.txt', mode='w')
    file2 = open(args.dataset_str + '_roc_curr.txt', mode='w')
    file3 = open(args.dataset_str + '_ap_curr.txt', mode='w')'''
    hidden_emb = None
    for epoch in range(args.epochs):
        if epoch< 80:
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
            # tr_asr = train_asr(hidden_emb, adj_orig, num_back_edge, back_nodes_0, back_nodes_1)

            '''file1.write("%f" % cur_loss + '\n')
            file2.write("%f" % roc_curr + '\n')
            file3.write("%f" % ap_curr + '\n')'''
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))
            # print("train_asr=", "{:.5f}". format(tr_asr))
        else:
            if (epoch%20==0)and(epoch<100):
                for i in range(pso_sizepop):
                    features_array = features.numpy()

                    features_array[n_nodes - 1] = pop[i][0]
                    features_array[n_nodes - 2] = pop[i][1]
                    features = torch.from_numpy(features_array)
                    recovered, mu, logvar = model(features, pso_adj_norm)
                    # hidden_emb=mu.data.numpy()
                    # tr_asr = train_asr(hidden_emb, adj_orig,0, num_back_edge_train, back_nodes_0,
                    #                    back_nodes_1)
                    # fitness[i]=tr_asr

                    pso_preds=back_pred(recovered,pso_trigger_node_lists_train)
                    loss = pso_loss(pso_preds, pso_back_lable, pso_norm, None)
                    fitness[i]=loss

                gbestpop, gbestfitness = pop[fitness.argmin()].copy(), fitness.min()
                pbestpop, pbestfitness = pop.copy(), fitness.copy()


                for i in range(pso_maxgen):
                    t = 0.5
                    # 速度更新
                    # print("ep",'%04d' % (i+ 1),"train_loss=", "{:.5f}".format(gbestfitness), gbestpop,pop[8])
                    print("ep", '%04d' % (i + 1), "train_loss=", "{:.5f}".format(gbestfitness),gbestpop)
                    for j in range(pso_sizepop):
                        v[j][0] += pso_lr[0] * np.random.rand() * (pbestpop[j][0] - pop[j][0]) + pso_lr[
                            1] * np.random.rand() * (gbestpop[0] - pop[j][0])
                        v[j][1] += pso_lr[0] * np.random.rand() * (pbestpop[j][1] - pop[j][1]) + pso_lr[
                            1] * np.random.rand() * (gbestpop[1] - pop[j][1])

                        # v[j][0] = pso_lr[0] * np.random.rand() * (pbestpop[j][0] - pop[j][0]) + pso_lr[
                        #     1] * np.random.rand() * (gbestpop[0] - pop[j][0])
                        # # print('第', j, 'v1', v[j][0])
                        # v[j][1] = pso_lr[0] * np.random.rand() * (pbestpop[j][1] - pop[j][1]) + pso_lr[
                        #     1] * np.random.rand() * (gbestpop[1] - pop[j][1])
                    v[v < pso_rangespeed[0]] = pso_rangespeed[0]
                    v[v > pso_rangespeed[1]] = pso_rangespeed[1]

                    # 粒子位置更新
                    for j in range(pso_sizepop):
                        # pop[j] += 0.5*v[j]
                        pop[j][0] = (0.5 * v[j][0]) +  pop[j][0]
                        # print('第', j, 'pop', pop[j][0])
                        pop[j][1] = (0.5 * v[j][1]) +  pop[j][1]
                    pop[pop < pso_rangepop[0]] = pso_rangepop[0]
                    pop[pop > pso_rangepop[1]] = pso_rangepop[1]


                    # 适应度更新
                    for j in range(pso_sizepop):
                        features_array = features.numpy()

                        features_array[n_nodes - 1] = pop[j][0]
                        features_array[n_nodes - 2] = pop[j][1]

                        # pop[j][0] = np.where(pop[j][0]>0.5, 1,0)
                        # pop[j][1] = np.where(pop[j][1] > 0.5, 1,0)
                        # features_array[n_nodes - 1] = np.where(pop[j][0]>0.5, 1,0)
                        # features_array[n_nodes - 2] = np.where(pop[j][1] > 0.5, 1,0)

                        features = torch.from_numpy(features_array)
                        recovered, mu, logvar = model(features, pso_adj_norm)
                        pso_preds = back_pred(recovered, pso_trigger_node_lists_train)
                        loss = pso_loss(pso_preds, pso_back_lable, pso_norm, None)
                        fitness[j] = loss

                        # hidden_emb = mu.data.numpy()
                        # tr_asr = train_asr(hidden_emb, adj_orig, 0, num_back_edge_train, back_nodes_0,
                        #                    back_nodes_1)
                        # fitness[i] = tr_asr

                    for j in range(pso_sizepop):
                        if fitness[j] < pbestfitness[j]:
                            pbestfitness[j] = fitness[j]
                            pbestpop[j] = pop[j].copy()

                    if pbestfitness.min() < gbestfitness:
                        gbestfitness = pbestfitness.min()
                        gbestpop = pop[pbestfitness.argmin()].copy()


                features_array = features.numpy()

                features_array[n_nodes - 1] = gbestpop[0]
                features_array[n_nodes - 2] = gbestpop[1]
                features = torch.from_numpy(features_array)


            t = time.time()
            recovered, mu, logvar = model_2(features, adj_norm_train)
            pso_preds = back_pred(recovered, pso_trigger_node_lists_train)
            loss = pso_loss(pso_preds, pso_back_lable, pso_norm, None)
            print('---------', loss)


            model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = model(features, adj_norm_train)
            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.numpy()
            roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
            # tr_asr = train_asr(hidden_emb, adj_orig, num_back_edge, back_nodes_0, back_nodes_1)

            '''file1.write("%f" % cur_loss + '\n')
            file2.write("%f" % roc_curr + '\n')
            file3.write("%f" % ap_curr + '\n')'''
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))
            # print("train_asr=", "{:.5f}". format(tr_asr))



    '''file1.close()
    file2.close()
    file3.close()'''
    print("Optimization Finished!Total epoch: 125")
    # 有了ROC怎么得到AUC
    roc_score, ap_score, _, _ = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print(args.dataset_str + ' test ROC score: ' + str(roc_score))
    print(args.dataset_str + ' test AP score: ' + str(ap_score))
    recovered, mu, logvar = model(features, adj_norm_test)

    hidden_emb = mu.data.numpy()
    #adj_rec = np.dot(hidden_emb, hidden_emb.T)
    #print(adj_rec[0,1378])
    #edge_list = edge_not(adj_rec)
    #edge_list_test = pd.DataFrame(data=edge_list)
    #np.savetxt('pred_1.csv', edge_list_test, fmt='%d')


    tr_asr = train_asr(hidden_emb, adj_orig, num_back_edge_train,num_back_edge_test, back_nodes_0, back_nodes_1)
    print("train_asr=", "{:.5f}".format(tr_asr))

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
