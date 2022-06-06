#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:59:01 2022

@author: edward
"""

import numpy as np
import pandas as pd

import networkx as nx


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def load_data(name):
#drug original dataset
    if(name=="drug"):
        drug=pd.read_csv('DRUGNET.csv',header=None)
        A=np.array(drug)
        G=nx.from_numpy_matrix(A)
        G_undirect=G.to_undirected()
        #convert directed graph to unditrected graph
        #find biggest connected subgraph
        sub_drug=max(nx.connected_components(G_undirect),key=len)
        sub_drug=list(sub_drug)
        #extrat sensitive attributes(gender)
        drug_attr=pd.read_csv('DRUGATTR.csv',header=None)
        drug_attr=drug_attr.iloc[1:,2:3]
        drug_attr=drug_attr.astype('int32')
        drug_attr=np.array(drug_attr).reshape(-1,1)
        #sensitive=drug_attr(sub_drug+1)
        sensitive=drug_attr[sub_drug]
        list_sensitive=[]
        for item in sensitive:
            list_sensitive.append(int(item))
            
        A = nx.adjacency_matrix(G_undirect)
        adja=A.todense()
        adja=adja[sub_drug]
        adja=adja[:,sub_drug]
        sensitive=np.array(list_sensitive)
        sensitive[sensitive==0]=1
        sensitive=sensitive-1
    
    return adja,sensitive
