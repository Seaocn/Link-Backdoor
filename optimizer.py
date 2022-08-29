import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np



def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def loss_function_GNAE(preds, labels, norm, pos_weight):

    cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    return cost


def pso_loss(preds, labels, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return  cost

def back_pred(adj,back_list):
    x=len(back_list)
    adj_array=adj.detach().numpy()
    list=np.zeros((x))
    for i in range (x):
        list[i]=adj_array[back_list[i][0]][back_list[i][1]]
    list=torch.from_numpy(list)
    return list


def dc_loss(n_true,n_false):
    true_label=torch.ones_like(n_true)
    false_label=torch.zeros_like(n_false)
    dc_true_loss=F.binary_cross_entropy_with_logits(n_true,true_label)
    dc_false_loss= F.binary_cross_entropy_with_logits(n_false,false_label)
    return dc_false_loss+dc_true_loss

def generator_loss(n_false,preds, labels, norm, pos_weight):
    true_label = torch.ones_like(n_false)
    generator_loss_false=F.binary_cross_entropy_with_logits(n_false,true_label)
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return cost+generator_loss_false

def generator_loss_VARGA(n_false):
    true_label = torch.ones_like(n_false)
    generator_loss_false=F.binary_cross_entropy_with_logits(n_false,true_label)

    return generator_loss_false

