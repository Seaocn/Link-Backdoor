# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import torch


def getweight():
    # 惯性权重
    weight = 1
    return weight

def getlearningrate():
    # 分别是粒子的个体和社会的学习因子，也称为加速常数
    lr = (0.49445,1.49445)
    # lr = (1, 0.5)
    # lr=(1,0.5)
    return lr

def getmaxgen():
    # 最大迭代次数
    maxgen = 22
    return maxgen

def getsizepop():
    # 种群规模
    sizepop = 50
    return sizepop

def getrangepop():
    # 粒子的位置的范围限制,x、y方向的限制相同
    rangepop = (0 , 1)
    #rangepop = (-2,2)
    return rangepop

def getrangespeed():
    # 粒子的速度范围限制
    rangespeed = (-20,20)
    return rangespeed

def func(x):
    # x输入粒子位置
    # y 粒子适应度值
    if (x[0]==0)&(x[1]==0):
        y = np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2)-2.71289
    else:
        y = np.sin(np.sqrt(x[0]**2+x[1]**2))/np.sqrt(x[0]**2+x[1]**2)+np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2)-2.71289
    return y

def initpopvfit(sizepop,fea_dim,nodes):
    pop = np.zeros((sizepop, fea_dim,2))
    v = np.zeros((sizepop,fea_dim,2))
    fitness = np.zeros(sizepop)

    for i in range(sizepop):
        pop[i][0] = np.random.randint(0,2,size=fea_dim,dtype=int)
        pop[i][1] = np.random.randint(0,2,size=fea_dim,dtyoe=int)
        v[i][0] =  np.random.randint(0,2,size=fea_dim,dtype=int)
        v[i][1] =  np.random.randint(0,2,size=fea_dim,dtype=int)
        fitness[i] = func(pop[i])
    return pop,v,fitness

def getinitbest(fitness,pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop,gbestfitness = pop[fitness.argmax()].copy(),fitness.max()
    #个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
    pbestpop,pbestfitness = pop.copy(),fitness.copy()

    return gbestpop,gbestfitness,pbestpop,pbestfitness

