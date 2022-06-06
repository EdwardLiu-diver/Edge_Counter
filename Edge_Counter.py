#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:11:48 2022

@author: edward
"""
import numpy as np

def Edge_Counter(sensitive,labels,adja):
    #k is the number of clusters
    k=len(np.unique(labels))
    #h is the number of sensitive groups
    h=len(np.unique(sensitive))
    
    #store each cluster node index in cluster_agg
    cluster_agg=[]
    for j in range(k):
        cluster=[]
        for i in range(len(labels)):
            if(labels[i]==j):
                cluster.append(i)
        cluster_agg.append(cluster)
    #special case , h=2, boy and girl
    #split the sensitive in each cluster
    sensitive_split=[]
    sen_count_cluster=[]
    for item in cluster_agg:
        boy,girl=split_sen(item, sensitive)
        sen_count={'boy':len(boy),'girl':len(girl),'ratio_boy2girl':len(boy)/(len(girl)+0.00001),'ratio_girl2boy':len(girl)/(len(boy)+0.00001)}
        dictionay={'boy':boy,'girl':girl}
        sensitive_split.append(dictionay)
        sen_count_cluster.append(sen_count)
    #
    #
        
        
    edge_fair=[]
    edges_each_cluster=[]
    for item in sensitive_split:
        boy=item['boy']
        num_boy=len(boy)
        girl=item['girl']
        num_girl=len(girl)
        boy_inner,girl_inner,inter=count_edges_cluster(boy,girl,adja)
        edge_dict={'boy_inner':boy_inner,'girl_inner':girl_inner,'sensitive_inter':inter}
        
        edge_fair_dict={'sum':boy_inner+girl_inner+inter,'ratio_boy2inter':boy_inner/(inter+0.001),
                        'ratio_gril2inter':girl_inner/(inter+0.001),'edge_fair':inter/(num_boy+num_girl)}
        edges_each_cluster.append(edge_dict)
        edge_fair.append(edge_fair_dict)
        
    
    
    
    return edges_each_cluster,sen_count_cluster,edge_fair
        
        
        
        
   
        
        
        
def split_sen(cluster,sensitive):
    boy=[]
    girl=[]
    for item in cluster:
        if sensitive[item]==0:
            boy.append(item)
        else:
            girl.append(item)
    return boy,girl 
        
    
        
def count_edges_cluster(boy_group,girl_group,adja):
    boy_inner=inner_edges(boy_group,boy_group,adja)
    girl_inner=inner_edges(girl_group,girl_group,adja)
    inter=inter_edges(boy_group,girl_group,adja)
    return boy_inner,girl_inner,inter
    

def inner_edges(group0,group1,adja):
    count=0
    for item in group0:
        for item2 in group1:
            if(adja[item,item2]==1):
                count+=1
    return count/2


def inter_edges(group0,group1,adja):
    count=0
    for item in group0:
        for item2 in group1:
            if(adja[item,item2]==1):
                count+=1
    return count
    
    
    