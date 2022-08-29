# link_backdoor
This is a PyThorh implementation of Backdoor Attack on Link Prediction via Node Injection, as described in our paper:

Haibin Zheng, Haiyang Xiong, Haonan Ma, Guohan Huang, Jinyin Chen, [Link-Backdoor: Backdoor Attack on Link Prediction via Node Injection](https://doi.org/10.48550/arXiv.2208.06776)
 


## Step -1: Requirement

The code requires Python >=3.6 and is built on PyTorch. Note that PyTorch may need to be [installed manually](https://pytorch.org/get-started/locally/) depending on different platforms and CUDA drivers.

## Step 0: Datasets

We provide the datasets used in our paper:

```bash
[ "cora","cora_ML" ,"citeseer","pubmed","CS"]
```

## Step 1: Preparation

Find the links for attack training
```bash
python find_link.py --model VGAE --dataset_str cora --hidden1 32 \
--hidden2 16 --dropout 0.1 --lr 0.01
```

## Step 2: Attack

Training the link_backdoor model
```bash
python main.py --model VGAE --dataset_str cora --hidden1 32 \
--hidden2 16 --dropout 0.1 --lr 0.01 --attalink 540 --alllink 876
```