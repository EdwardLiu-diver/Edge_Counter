#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:03:34 2022

@author: edward
"""


from load_data import load_data
from Edge_Counter import Edge_Counter
from model import unnormalized_fair_spectral
from model import unnormalized_spectral
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from oka_unc import load_oka
from oka_unc import load_unc





#drug net 
# adja,sensitive=load_data()
# G=nx.from_numpy_matrix(adja)
#labels_fair,intertia,H_fair=unnormalized_fair_spectral(G, 10, sensitive, random_state=1)

#unc
adja,sensitive,G=load_unc()

#adja=adja.to_dense()

balance_spectral=[]
balance_fair=[]
std_edge_spectral=[]
std_edge_fair=[]


def averge_balance(sen_count_cluster):
    sumcount=0
    for item in sen_count_cluster:
        sumcount+=min(item['ratio_boy2girl'],item['ratio_girl2boy'])
    return sumcount/len(sen_count_cluster)


def std_edgeFair(edge_fair):
    std_edgeFair=[]
    for item in edge_fair:
        std_edgeFair.append(item['edge_fair'])
    std_edgeFair=np.array(std_edgeFair)
    return np.std(std_edgeFair)

for i in range(1,10):
    labels_fair,intertia,H_fair=unnormalized_fair_spectral(G, i, sensitive, random_state=1)
    labels_spectral,intertia_spectral=unnormalized_spectral(G,i,random_state=1)
    
    fair_edges_each_cluster,fair_sen_count_cluster,fair_edge_fair=Edge_Counter(sensitive, labels_spectral, adja)
    edges_each_cluster,sen_count_cluster,edge_fair=Edge_Counter(sensitive, labels_fair, adja)
    
    balance_fair_iter=averge_balance(fair_sen_count_cluster)
    balance_spectral_iter=averge_balance(sen_count_cluster)
    
    std_edge_fair_iter=std_edgeFair(fair_edge_fair)
    std_edge_spectral_iter=std_edgeFair(edge_fair)
    
    
    std_edge_spectral.append(std_edge_spectral_iter)
    std_edge_fair.append(std_edge_fair_iter)
    
    
    balance_fair.append(balance_fair_iter)
    balance_spectral.append(balance_spectral_iter)
    
    




    
plt.figure()

x=np.arange(9)
plt.plot(x,std_edge_fair,'b',label='fairSC')
plt.plot(x,std_edge_spectral,'r',label='SC')
plt.title('Std of fair notion')
plt.legend()
plt.show()

    
plt.figure()

x=np.arange(9)
plt.plot(x,balance_fair,'b',label='fairSC')
plt.plot(x,balance_spectral,'r',label='SC')
plt.title('Average Balance')
plt.legend()
plt.show()

    



# node2vec = Node2Vec(G,dimensions=193, walk_length = 10, num_walks = 10,p = 1, q = 1, workers = 4)#init model
# model=node2vec.fit(window = 5,min_count=0)# train model
# model.wv.save_word2vec_format(u'embedding.txt')




