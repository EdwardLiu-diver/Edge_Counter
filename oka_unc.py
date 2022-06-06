#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:45:48 2022

@author: edward
"""


import scipy.io
import networkx as nx

def detect_zero(gender):
    node_list=[]
    for i in range(len(gender)):
        if(gender[i]==0):
            node_list.append(i)
    return node_list


def load_oka():
    mat = scipy.io.loadmat('Oklahoma97.mat')
    sparse=mat['A']
    sensitive=mat['local_info']
    gender=sensitive[:,2]
    G=nx.from_scipy_sparse_matrix(sparse)
    
    node_list=detect_zero(gender)
    G.remove_nodes_from(node_list)
    
    largest=list(max(nx.connected_components(G), key=len))
    
    H=G.subgraph(largest)
    
    gender_sub=gender[H.node]
    adja=nx.adjacency_matrix(H)
    return adja,gender_sub,H
    
    
def load_unc():
    mat = scipy.io.loadmat('UNC28.mat')
    sparse=mat['A']
    sensitive=mat['local_info']
    gender=sensitive[:,2]
    G=nx.from_scipy_sparse_matrix(sparse)

    
    node_list=detect_zero(gender)
    
    G.remove_nodes_from(node_list)
    
    largest=list(max(nx.connected_components(G), key=len))
    
    H=G.subgraph(largest)
    
    gender_sub=gender[H.node]
    adja=nx.adjacency_matrix(H)
    return adja,gender_sub,H

