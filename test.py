# coding=utf-8
from __future__ import division
from __future__ import print_function
import random
import train
import numpy as np
import scipy.sparse as sp
import torch
import argparse
from gae.model import GCNModelVAE
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score

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
def rest_n(back_e, adjtrain, n_b_e, subg, test_e_f):
    # back_edges即从test_edge中选出目标连边对应的节点
    rest_n = []
    while len(rest_n) < (n_b_e * (len(subg) - 2)):
        d1 = random.randint(0, len(adjtrain) - 1)
        d2 = random.randint(0, len(adjtrain) - 1)
        if d1 != d2:
            if (d1 not in rest_n) and (d1 not in back_e) and (d2 not in rest_n) and (d2 not in back_e) and\
                    ([d1, d2] not in test_e_f) and ([d2, d1] not in test_e_f):
                rest_n.append(d1)
                rest_n.append(d2)
            else:
                continue
        else:
            continue
    print("Complete rest nodes")
    return rest_n
'''


def inject_test(adj_t, trigger_node_list, n_b_e, subg):
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        h = 1
        for j in range(len(subg)-1):
            for k in range(h, len(subg)):
                adj_t[jd[j]][jd[k]] = adj_t[jd[k]][jd[j]] = subg[j][k]
            h += 1
    print("Injection finished")


def get_asr(emb, adj_orig, n_b_e, tar_nodes_0, tar_nodes_1):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    possible = []
    for i in range(n_b_e):
        pos.append(adj_orig[tar_nodes_0[i], tar_nodes_1[i]])
        possible.append(sigmoid(adj_rec[tar_nodes_0[i], tar_nodes_1[i]]))
        if sigmoid(adj_rec[tar_nodes_0[i], tar_nodes_1[i]]) > 0.5:
            preds.append(1)
        else:
            preds.append(0)

    count = 0
    for i in range(len(preds)):
        if preds[i] != pos[i]:
            count += 1
        else:
            continue
    ASR = count/len(preds)

    return ASR, possible

'''
def get_false_nodes(emb):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    ll = len(adj_rec)
    list_false_nodes0 = []
    list_false_nodes1 = []
    for i in range(ll):
        for j in range(ll):
            if sigmoid(adj_rec[i][j]) < 0.3:
                list_false_nodes0.append(i)
                list_false_nodes1.append(j)
    return list_false_nodes0, list_false_nodes1
'''


def tuijian(emb, adj_orig, listn):
    print(listn)
    n1 = listn[0]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    adj_rec_n1 = adj_rec[n1]
    possible = []
    for i in range(len(adj_rec_n1)):
        possible.append(sigmoid(adj_rec_n1[i]))
    # possible.sort(reverse=True)#can not use this
    for l in range(len(possible)):
        if (adj_orig[n1][l] == 1) or (l == n1):
            possible[l] = 0
    re_list = []

    sorted_nums = sorted(enumerate(possible), key=lambda x: x[1])
    idx = [j[0] for j in sorted_nums]
    nums = [k[1] for k in sorted_nums]

    for m in range(1, 21):
        re_list.append(idx[-m])
    # print(nums[-1])
    # print(nums[-20])
    print("recommendation list:")
    print(re_list)
    if listn[3] in re_list:
        print(re_list.index(listn[3]) + 1)


def gongji(n1, n4, adjtrain, subg):
    list_n = []
    list_n.append(n1)
    while len(list_n) < 2:
        n2 = random.randint(0, len(adjtrain) - 1)
        if (n2 != n1) and (n2 != n4):
            list_n.append(n2)
        else:
            continue
    while len(list_n) < 3:
        n3 = random.randint(0, len(adjtrain) - 1)
        if (n3 != n1) and (n3 != n4):
            list_n.append(n3)
        else:
            continue
    list_n.append(n4)

    h = 1
    for j in range(len(subg) - 1):
        for k in range(h, len(subg)):
            adjtrain[list_n[j]][list_n[k]] = adjtrain[list_n[k]][list_n[j]] = subg[j][k]
        h += 1
    print("Attack finished")

    return list_n, adjtrain


def get_possibility(emb, n1, n2):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    return sigmoid(adj_rec[n1][n2])


def main():#nn1, nn4
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj_tr = adj_train.toarray()
    adj_ori = adj_orig.toarray()
    '''
    adj_tr = np.loadtxt(args.dataset_str + '_adj_tr.csv', unpack=False)
    test_edges = np.loadtxt(args.dataset_str + '_test_edges.csv',  dtype=int, unpack=False)
    t_e_f = np.loadtxt(args.dataset_str + '_test_edges_false.csv',  dtype=int, unpack=False)
    test_edges_false = t_e_f.tolist()
    '''

    # num_back_edge = 100
    # sub_g = np.loadtxt('subg_4_83.csv', unpack=False)
    #print("number of injected-graph: %d" % num_back_edge)
    #target_nodes, target_nodes_0, target_nodes_1 = train.tar_edges(test_edges, num_back_edge)
    '''
    target_nodes_0 = np.loadtxt(args.dataset_str + '_tar/right_false_node0.csv', dtype=int, unpack=False)
    target_nodes_1 = np.loadtxt(args.dataset_str + '_tar/right_false_node1.csv', dtype=int, unpack=False)
    # target_nodes_0 = np.loadtxt('right_false_node0.csv', dtype=int, unpack=False)
    # target_nodes_1 = np.loadtxt('right_false_node1.csv', dtype=int, unpack=False)
    target_nodes = []
    for p in range(len(target_nodes_0)):
        target_nodes.append(target_nodes_0[p])
        target_nodes.append(target_nodes_1[p])
    num_back_edge = len(target_nodes_0)

    rest_node = train.rest_nodes(target_nodes, adj_tr, num_back_edge, sub_g)
    trigger_node_lists = train.trigger_nodes(target_nodes, rest_node, num_back_edge, sub_g)
    inject_test(adj_tr, trigger_node_lists, num_back_edge, sub_g)
    '''
    # list_n, adj_tr = gongji(nn1, nn4, adj_tr, sub_g)
    adj_tr = sp.csr_matrix(adj_tr)
    adj = adj_tr
    adj_norm = preprocess_graph(adj)

    # need to initiaize a model
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model.load_state_dict(torch.load(args.dataset_str + "_clean_model.pth"))
    # model.load_state_dict(torch.load("300_" + args.dataset_str + "_backdoor_model.pth"))
    #model.load_state_dict(torch.load("200_" + args.dataset_str + "_backdoor_model.pth"))
    # model.load_state_dict(torch.load("2000_" + args.dataset_str + "_backdoor_model.pth"))
    model.eval()

    recovered, mu, logvar = model(features, adj_norm)
    hidden_emb = mu.data.numpy()
    #asr_score, poss = get_asr(hidden_emb, adj_orig, num_back_edge, target_nodes_0, target_nodes_1)
    #poss = get_possibility(hidden_emb, 106, 31)
    # roc_score, ap_score, right_false_node0, right_false_node1 = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    # print('Test ROC score: ' + str(roc_score))
    # print('Test AP score: ' + str(ap_score))

    # return roc_score, ap_score, right_false_node0, right_false_node1
    # return asr_score, poss
    return hidden_emb, adj_ori#, list_n

'''
if __name__ == '__main__':
    _, _, right_false_node0, right_false_node1 = main()
    np.savetxt('right_false_node02.csv', right_false_node0, fmt='%d')
    np.savetxt('right_false_node12.csv', right_false_node1, fmt='%d')
'''
'''
if __name__ == '__main__':
    A = []
    B = []
    h_emb = main()
    l_f_n0, l_f_n1 = get_false_nodes(h_emb)
    print(len(l_f_n0))
    while len(A) < 100:
        a = random.randint(0, len(l_f_n0))
        if (l_f_n0[a] not in A) and (l_f_n1[a] not in B):
            A.append(l_f_n0[a])
            B.append(l_f_n1[a])
        else:
            continue
    np.savetxt('right_false_node04.csv', A, fmt='%d')
    np.savetxt('right_false_node14.csv', B, fmt='%d')
'''
'''
if __name__ == '__main__':
    s = []
    target_nodes_0 = np.loadtxt(args.dataset_str + '_tar/right_false_node0.csv', dtype=int, unpack=False)
    for _ in range(5):
        asr, poss = main()
        s.append(asr)
        de = []
        a = []
        for i in range(len(poss)):
            if poss[i] > 0.5:
                de.append(target_nodes_0[i])
                a.append(i)
        print(de)
        print(a)
    print(s)
    final_s = sum(s) / 5
    print(final_s)
'''
'''
if __name__ == '__main__':
    s = []
    for _ in range(5):
        asr, poss = main()
        s.append(asr)
    print(s)
    final_s = sum(s) / 5
    print(args.dataset_str + " test ASR:" + str(final_s))
'''
'''
if __name__ == '__main__':
    r = []
    a = []
    for _ in range(5):
        roc, ap, _, _ = main()
        r.append(roc)
        a.append(ap)
    final_r = sum(r) / 5
    print(final_r)
    final_a = sum(a) / 5
    print(final_a)
'''

if __name__ == '__main__':
    h_emb, adj_ori = main()
    np.savetxt(args.dataset_str + '_adj.csv', adj_ori, fmt='%d')
    '''
    print('请输入当前被搜索到的论文编号：')
    n1 = int(input())
    print('请输入目标论文编号：')
    n4 = int(input())
    for i in range(5):
        h_emb, adj_ori, list_n = main(n1, n4)
        tuijian(h_emb, adj_ori, list_n)
    '''