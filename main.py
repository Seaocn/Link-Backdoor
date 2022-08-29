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


from model import GCNModelVAE,GIC,GAE,ARGA,discriminator,ARVGA
from optimizer import loss_function,back_pred,pso_loss,loss_function_GNAE,dc_loss,generator_loss,generator_loss_VARGA
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score,sparse_mx_to_torch_sparse_tensor_GNAE,GBA_LOSS,find_gard_max,get_AMC_score,load_data_2
import os

#设置cpu占用
num_threads = 1
# os.environ ['OMP_NUM_THREADS'] = str(num_threads)
# os.environ ['OPENBLAS_NUM_THREADS'] = str(num_threads)
# os.environ ['MKL_NUM_THREADS'] = str(num_threads)
# os.environ ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
# os.environ ['NUMEXPR_NUM_THREADS'] = str(num_threads)
torch.set_num_threads(num_threads)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
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
parser.add_argument('--attalink',type=int,default=8189,help='540,470,4430,430,8189')
parser.add_argument('--alllink',type=int,default=11063,help='876,960,8305,706,11063')
args = parser.parse_args()







def trigger_nodes_not_subg(back_nodes, n_b_e):
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


def add_node(adj,features):
    n_nodes, feat_dim = features.shape
    csr_matrix1=np.zeros((n_nodes,1))
    csr_matrix2=np.zeros((1,n_nodes+1))
    csr_matrix1=sp.csr_matrix(csr_matrix1)
    csr_matrix2=sp.csr_matrix(csr_matrix2)
    adj=sp.hstack((adj,csr_matrix1))
    adj=sp.vstack((adj,csr_matrix2))
    adj=adj.A
    adj=sp.csr_matrix(adj)
    t=time.time()

    features_array=features.numpy()
    index=np.random.randint(0,n_nodes)
    features_new_node=features_array[index]
    features_new_node=sp.csr_matrix(features_new_node)
    features_array=sp.csr_matrix(features_array)
    t=time.time()-t
    print(2*t)



    features=sp.vstack((features_array,features_new_node))
    features=features.A
    features=torch.from_numpy(features)

    return adj,features

def inject_lables_add2_node(adj_t, trigger_node_list, n_b_e, subg):
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        adj_t[adj_t.shape[0] - 2][jd[0]] = adj_t[jd[0]][adj_t.shape[0] - 2] = 1
        adj_t[adj_t.shape[0] - 2][jd[1]] = adj_t[jd[1]][adj_t.shape[0] - 2] = 1
        adj_t[adj_t.shape[0] - 1][jd[0]] = adj_t[jd[0]][adj_t.shape[0] - 1] = 1
        adj_t[adj_t.shape[0] - 1][jd[1]] = adj_t[jd[1]][adj_t.shape[0] - 1] = 1
        adj_t[jd[0]][jd[1]] = adj_t[jd[1]][jd[0]] = 1
    adj_t[adj_t.shape[0] - 1][adj_t.shape[0] - 2] = adj_t[adj_t.shape[0] - 2][adj_t.shape[0] - 1] = 1
    print("Injection train finished")

def inject_edge2_node(adj_t, trigger_node_list, n_b_e, subg):
    for i in range(n_b_e):
        jd = trigger_node_list[i]
        adj_t[adj_t.shape[0] - 2][jd[0]] = adj_t[jd[0]][adj_t.shape[0] - 2] = 1
        adj_t[adj_t.shape[0] - 2][jd[1]] = adj_t[jd[1]][adj_t.shape[0] - 2] = 1
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

def noise (features):
    n_nodes, feat_dim = features.shape
    z=int(feat_dim/10)
    noise=[]
    while len(noise)<z:
        n=np.random.randint(0,feat_dim)
        if n not in noise:
            noise.append(n)
    for i in range (z):
        features[n_nodes - 1][noise[i]]=1 if features[n_nodes-1][noise[i]]==0 else 0
        features[n_nodes - 2][noise[i]] = 1 if features[n_nodes - 2][noise[i]] == 0 else 0
    return features


def gae_for(args):

    print("Using {} dataset".format(args.dataset_str))
    if args.dataset_str == 'cora' or args.dataset_str == 'citeseer' or args.dataset_str == 'pubmed':
        adj, features = load_data(args.dataset_str)
    else:
        adj, features = load_data_2(args.dataset_str)
    adj, features = add_node(adj, features)  # 增加一个新的节点
    adj, features = add_node(adj, features)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # adj_train相比于adj_orig少了val_edges + test_edges的连边
    # train_edges没用上，val_edges、val_edges_false计算每个epoch的ROC和AP，test_edges、test_edges_false计算最终结果
    # adj_train class 'scipy.sparse.csr.csr_matrix'  _edges class 'numpy.ndarray'  _edges_false class 'list'
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # adj_train是要加入触发器的对象,转为array格式以便修改

    adj_label_orig = adj_train + sp.eye(adj_train.shape[0])  # eye()构造对角线为1的稀疏矩阵
    adj_l = adj_label_orig.toarray()
    adj_label_orig = torch.FloatTensor(adj_l)  # 训练初始模型要用到的lable

    adj_tr = adj_train.toarray()
    adj_norm = preprocess_graph(adj_tr)  # 训练初始模型用到的adj
    # adj_norm_GNAE = sparse_mx_to_torch_sparse_tensor_GNAE(adj_tr)

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
    num_back_edge = len(back_nodes_0)  # 后门攻击的连边数

    print("number of injected-graph: %d" % num_back_edge)
    sub_g = np.loadtxt('subg_4_67.csv', unpack=False)  # RBA会有用

    # rest_node = rest_nodes(target_nodes, adj_tr, num_back_edge, sub_g)#RBA
    # trigger_node_lists = trigger_nodes(target_nodes, rest_node, num_back_edge, sub_g)

    trigger_node_lists = trigger_nodes_not_subg(target_nodes, num_back_edge)
    pso_trigger_node_lists_train = trigger_node_lists[0:args.attalink:1]
    trigger_node_lists_train = trigger_node_lists[0:args.attalink:1]  # cora:54  citeseer:47   pubmed:443
    trigger_node_lists_test = trigger_node_lists[args.attalink:args.alllink:1]  # cora:876   citeseer:960   pubmed:8305
    num_back_edge_train = len(trigger_node_lists_train)
    num_back_edge_test = len(trigger_node_lists_test)

    pso_back_lable = np.ones((len(pso_trigger_node_lists_train)))  # 在粒子群计算损失度时会用到
    pso_back_lable = torch.from_numpy(pso_back_lable)
    pso_norm = float(len(pso_trigger_node_lists_train) / len(pso_trigger_node_lists_train))

    adj_tr = adj_train.toarray()

    inject_edge2_node(adj_tr, pso_trigger_node_lists_train, len(pso_trigger_node_lists_train), sub_g)
    pso_adj_train = sp.csr_matrix(adj_tr)
    pso_adj_norm = preprocess_graph(pso_adj_train)  # 粒子群优化的adj

    inject_edge2_node(adj_tr, trigger_node_lists_train, num_back_edge_train, sub_g)
    adj_train = sp.csr_matrix(adj_tr)
    adj = adj_train

    adj_tr_lable = adj_tr
    inject_lables_add2_node(adj_tr_lable, trigger_node_lists_train, num_back_edge_train, sub_g)
    # inject_train(adj_tr, trigger_node_lists_train, num_back_edge_train, sub_g)
    adj_train_lable = sp.csr_matrix(adj_tr_lable)  # 后门攻击的lable

    inject_edge2_node(adj_tr, trigger_node_lists_test, num_back_edge_test, sub_g)
    # inject_test(adj_tr, trigger_node_lists_test, num_back_edge_test, sub_g)
    adj_test = sp.csr_matrix(adj_tr)

    # print(trigger_node_lists_train)
    # print(trigger_node_lists_test)
    # Some preprocessing 一些预处理,adj被改变后adj_norm、adj_label也相应会改变
    adj_norm_train = preprocess_graph(adj)  # class 'torch.Tensor'
    adj_norm_train_GNAE = sparse_mx_to_torch_sparse_tensor_GNAE(adj)
    adj_norm_test = preprocess_graph(adj_test)
    adj_norm_test_GNAE = sparse_mx_to_torch_sparse_tensor_GNAE(adj_test)

    # adj_train = sp.csr_matrix(adj_tr)
    # adj = adj_train
    # adj_norm = preprocess_graph(adj)  # class 'torch.Tensor'

    adj_label = adj_train_lable + sp.eye(adj_train.shape[0])  # eye()构造对角线为1的稀疏矩阵
    # adj_label = sparse_to_tuple(adj_label)
    adj_l = adj_label.toarray()
    adj_label = torch.FloatTensor(adj_l)  # class 'torch.Tensor'

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # pos_weight = torch.FloatTensor(pos_weight)
    # pos_weight = tf.to_float(pos_weight)
    pos_weight = torch.tensor(pos_weight, dtype=float)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    model_2 = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    model_GIC = GIC(n_nodes, feat_dim, args.hidden1, args.num_clusters, args.dropout, args.beta).to(device)
    # model_GNAE= GNAE(feat_dim,128,args.dropout).to(device)
    # model_VGNAE = VGNAE(feat_dim, 32, args.dropout).to(device)
    model_GAE = GAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    model_ARGA = ARGA(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    model_dc = discriminator(args.hidden1, args.hidden2, args.hidden3).to(device)
    model_dc_2 = discriminator(args.hidden1, args.hidden2, args.hidden3).to(device)
    model_ARVGA = ARVGA(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=args.lr)
    optimizer_GIC = optim.Adam(model_GIC.parameters(), lr=args.lr, weight_decay=0.0)
    # optimizer_GNAE = optim.Adam(model_GNAE.parameters(), lr=0.005)
    # optimizer_VGNAE=optim.Adam(model_VGNAE.parameters(), lr=0.005)
    optimizer_GAE = optim.Adam(model_GAE.parameters(), lr=args.lr)
    optimizer_ARGA = optim.Adam(model_ARGA.parameters(), lr=args.lr)
    optimizer_ARGA_2 = optim.Adam(model_ARGA.parameters(), lr=args.lr)
    optimizer_DC = optim.Adam(model_dc.parameters(), lr=args.lr)
    optimizer_DC_2 = optim.Adam(model_dc_2.parameters(), lr=args.lr)
    optimizer_ARVGA = optim.Adam(model_ARVGA.parameters(), lr=args.lr)
    optimizer_ARVGA_2 = optim.Adam(model_ARVGA.parameters(), lr=args.lr)

    b_xent = torch.nn.BCEWithLogitsLoss()
    b_bce = torch.nn.BCELoss()
    GIC_best = 1e9

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
        # pop[i][0]= np.random.rand(feat_dim)
        # pop[i][1] =np.random.rand(feat_dim)
        # v[i][0] = np.random.rand(feat_dim)
        # v[i][1] = np.random.rand(feat_dim)
        pop[i][0] = np.random.randint(0, 2, size=feat_dim)
        pop[i][1] = np.random.randint(0, 2, size=feat_dim)
        v[i][0] = np.random.randint(0, 2, size=feat_dim)
        v[i][1] = np.random.randint(0, 2, size=feat_dim)

    pop.astype(np.float32)
    v.astype(np.float32)

    hidden_emb = None
    for epoch in range(args.epochs):
        if epoch< args.preepochs:
            t = time.time()

            if args.model=='VGAE':
                model.train()
                optimizer.zero_grad()
                recovered, mu, logvar = model(features, adj_norm)
                loss = loss_function(preds=recovered, labels=adj_label_orig,
                                     mu=mu, logvar=logvar, n_nodes=n_nodes,
                                     norm=norm, pos_weight=pos_weight)
                # recovered, mu, logvar = model(features, adj_norm_train)
                # loss = loss_function(preds=recovered, labels=adj_label,
                #                      mu=mu, logvar=logvar, n_nodes=n_nodes,
                #                      norm=norm, pos_weight=pos_weight)

                loss.backward()
                cur_loss = loss.item()
                optimizer.step()

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

                cur_loss=loss
                optimizer_GIC.step()
                print(epoch)






            elif args.model=='GAE':
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
            elif args.model=='ARGA':
                model_ARGA.train()
                model_dc.train()
                n_true=torch.rand(n_nodes,args.hidden2)
                optimizer_DC.zero_grad()
                optimizer_ARGA.zero_grad()
                optimizer_ARGA_2.zero_grad()
                recovered,mu =model_ARGA(features,adj_norm)
                loss=loss_function_GNAE(preds=recovered, labels=adj_label_orig, norm=norm, pos_weight=pos_weight)
                if epoch%5==0:
                   n_true=model_dc(n_true)
                   n_false=model_dc(mu)
                   loss_dc=dc_loss(n_true,n_false)
                   loss_ge=generator_loss(n_false,preds=recovered, labels=adj_label_orig, norm=norm, pos_weight=pos_weight)
                loss.backward(retain_graph=True)
                if epoch%5==0:
                   loss_dc.backward(retain_graph=True)
                   loss_ge.backward(retain_graph=True)
                optimizer_ARGA_2.step()
                if epoch%5==0:
                   optimizer_DC.step()
                   optimizer_ARGA.step()
                cur_loss=loss.item()

                # optimizer_ARGA.step()
                hidden_emb = mu.data.numpy()
                roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)


                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "val_ap=", "{:.5f}".format(ap_curr),
                      "time=", "{:.5f}".format(time.time() - t))
            elif args.model=='ARVGA':
                model_ARVGA.train()
                model_dc.train()
                n_true = torch.rand(n_nodes, args.hidden2)
                optimizer_DC.zero_grad()
                optimizer_ARVGA.zero_grad()
                optimizer_ARVGA_2.zero_grad()
                recovered, mu ,logvar= model_ARVGA(features, adj_norm)
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










        else:
            if (epoch%20==0)and(epoch<(80+args.preepochs)):

                t=time.time()
                features_var=torch.autograd.Variable(features,requires_grad=True)
                if args.model == 'VGAE':
                    recovered, mu, logvar = model(features_var, adj_norm_train)
                elif args.model == 'GAE':
                    recovered, mu = model_GAE(features_var, adj_norm_train)
                elif args.model == 'ARGA':
                    recovered,mu=model_ARGA(features_var,adj_norm_train)
                elif args.model == 'ARVGA':
                    recovered, mu ,logvar= model_ARVGA(features_var, adj_norm_train)
                elif args.model == 'GIC':
                    embeds, _, _, S = model_GIC.embed(features_var, adj_norm_train, None, args.beta)
                    embeds = embeds / embeds.norm(dim=1)[:, None]
                    recovered=torch.mm(embeds,embeds.t())

                loss_grad= GBA_LOSS (recovered,pso_trigger_node_lists_train,args.attalink)
                print('trigger loss1', loss_grad.item())
                gard=torch.autograd.grad(loss_grad,features_var,retain_graph=True)[0].data
                features=find_gard_max(gard,features,args.dataset_str)
                t_opt=time.time()-t
                print(t_opt)

            if args.model == 'VGAE':
                t = time.time()
                # recovered, mu, logvar = model_2(features, adj_norm_train)
                # pso_preds = back_pred(recovered, pso_trigger_node_lists_train)
                # loss = pso_loss(pso_preds, pso_back_lable, pso_norm, None)
                # print('---------', loss)
                #
                #
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

                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "val_ap=", "{:.5f}".format(ap_curr),
                      "time=", "{:.5f}".format(time.time() - t))
                # print("train_asr=", "{:.5f}". format(tr_asr))
            elif args.model == 'GIC':
                model_GIC.train()
                optimizer_GIC.zero_grad()

                idx = np.random.permutation(n_nodes)
                shuf_fts = features[idx, :]

                lbl_1 = torch.ones(1, n_nodes)
                lbl_2 = torch.zeros(1, n_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)

                logits, logits2 = model_GIC(features, shuf_fts, adj_norm_train, None, None, None, args.beta)
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

            elif args.model=='GAE':
                model_GAE.train()
                optimizer_GAE.zero_grad()
                recovered, mu = model_GAE(features, adj_norm_train)
                loss = loss_function_GNAE(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)

                loss.backward()
                cur_loss = loss.item()
                optimizer_GAE.step()

                hidden_emb = mu.data.numpy()
                roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "val_ap=", "{:.5f}".format(ap_curr),
                      "time=", "{:.5f}".format(time.time() - t))
            elif args.model == 'ARGA':
                model_ARGA.train()
                model_dc.train()
                n_true = torch.rand(n_nodes, args.hidden2)
                optimizer_DC.zero_grad()
                optimizer_ARGA.zero_grad()
                optimizer_ARGA_2.zero_grad()
                recovered, mu = model_ARGA(features, adj_norm_train)
                loss = loss_function_GNAE(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
                if epoch%5==0:
                   n_true=model_dc(n_true)
                   n_false=model_dc(mu)
                   loss_dc=dc_loss(n_true,n_false)
                   loss_ge=generator_loss(n_false,preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
                loss.backward(retain_graph=True)
                if epoch%5==0:
                   loss_dc.backward(retain_graph=True)
                   loss_ge.backward(retain_graph=True)
                optimizer_ARGA_2.step()
                if epoch%5==0:
                   optimizer_DC.step()
                   optimizer_ARGA.step()
                cur_loss = loss.item()
                hidden_emb = mu.data.numpy()
                roc_curr, ap_curr, _, _ = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "val_ap=", "{:.5f}".format(ap_curr),
                      "time=", "{:.5f}".format(time.time() - t))
            elif args.model == 'ARVGA':
                model_ARVGA.train()
                model_dc.train()
                n_true = torch.rand(n_nodes, args.hidden2)
                optimizer_DC.zero_grad()
                optimizer_ARVGA.zero_grad()
                optimizer_ARVGA_2.zero_grad()
                recovered, mu, logvar = model_ARVGA(features, adj_norm_train)
                loss = loss_function(preds=recovered, labels=adj_label,
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
                    loss_ge.backward()
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




    print("Optimization Finished!Total epoch: 200")
    # # 有了ROC怎么得到AUC
    roc_score, ap_score, _, _ = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print(args.dataset_str + ' test ROC score: ' + str(roc_score))
    print(args.dataset_str + ' test AP score: ' + str(ap_score))

    if args.model == 'VGAE':
        recovered, mu, logvar = model(features, adj_norm_test)
        hidden_emb = mu.data.numpy()

        tr_asr = train_asr(hidden_emb, adj_orig, num_back_edge_train, num_back_edge_test, back_nodes_0, back_nodes_1)
        AMC = get_AMC_score(hidden_emb, adj_orig, num_back_edge_train, num_back_edge_test, back_nodes_0, back_nodes_1)
        print("train_asr=", "{:.5f}".format(tr_asr), "train_AMC=", "{:.5f}".format(AMC))
    elif args.model == 'GAE':
        recovered, mu = model_GAE(features, adj_norm_test)
        hidden_emb = mu.data.numpy()

        tr_asr = train_asr(hidden_emb, adj_orig, num_back_edge_train, num_back_edge_test, back_nodes_0, back_nodes_1)
        AMC = get_AMC_score(hidden_emb, adj_orig, num_back_edge_train, num_back_edge_test, back_nodes_0, back_nodes_1)
        print("train_asr=", "{:.5f}".format(tr_asr), "train_AMC=", "{:.5f}".format(AMC))
    elif args.model == 'ARGA':
        recovered, mu = model_ARGA(features, adj_norm_test)
        hidden_emb = mu.data.numpy()

        tr_asr = train_asr(hidden_emb, adj_orig, num_back_edge_train, num_back_edge_test, back_nodes_0, back_nodes_1)
        AMC = get_AMC_score(hidden_emb, adj_orig, num_back_edge_train, num_back_edge_test, back_nodes_0, back_nodes_1)
        print("train_asr=", "{:.5f}".format(tr_asr), "train_AMC=", "{:.5f}".format(AMC))
    elif args.model == 'ARVGA':
        recovered, mu ,logvar= model_ARVGA(features, adj_norm_test)
        hidden_emb = mu.data.numpy()

        tr_asr = train_asr(hidden_emb, adj_orig, num_back_edge_train,num_back_edge_test, back_nodes_0, back_nodes_1)
        AMC = get_AMC_score(hidden_emb, adj_orig, num_back_edge_train,num_back_edge_test, back_nodes_0, back_nodes_1)
        print("train_asr=", "{:.5f}".format(tr_asr),"train_AMC=", "{:.5f}".format(AMC))
    elif args.model == 'GIC':
        embeds,_,_,S = model_GIC.embed(features, adj_norm, None, args.beta)
        embeds=embeds.detach()
        embeds = embeds / embeds.norm(dim=1)[:, None]
        roc_score, ap_score, _, _ = get_roc_score(embeds, adj_orig, val_edges, val_edges_false)
        print(args.dataset_str + ' test ROC score: ' + str(roc_score))
        print(args.dataset_str + ' test AP score: ' + str(ap_score))

        embeds,_,_,S= model_GIC.embed(features, adj_norm_test, None, args.beta)
        embeds=embeds.detach()
        embeds = embeds / embeds.norm(dim=1)[:, None]
        tr_asr = train_asr(embeds, adj_orig, num_back_edge_train, num_back_edge_test, back_nodes_0, back_nodes_1)
        AMC = get_AMC_score(embeds, adj_orig, num_back_edge_train,num_back_edge_test, back_nodes_0, back_nodes_1)
        print("train_asr=", "{:.5f}".format(tr_asr),"train_AMC=", "{:.5f}".format(AMC))




if __name__ == '__main__':


    gae_for(args)

