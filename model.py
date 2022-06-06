#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:02:47 2022

@author: edward
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from numpy.linalg import matrix_power
from numpy import inf
import networkx as nx
from scipy.linalg import null_space
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_array
from collections import Counter


def unnormalized_Spectral_Clustering(adja,k):

    # compute the Degree Matrix: D=sum(A)
    D = np.sum(adja, axis=1)
    # compute the Laplacian Matrix: L=D-A
    L = np.diag(D) - adja
    x, V = np.linalg.eig(L)
    x = x.real.astype(np.float32)
    V = V.real.astype(np.float32)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x:x[0])
    H = np.vstack([V[:,i] for (v, i) in x[:k]]).T
    sc_kmeans = KMeans(n_clusters=k).fit(H)
    
    return sc_kmeans.labels_


def normalized_Spectral_Clustering(adja,k):
    # compute the Degree Matrix: D=sum(A)
    D = np.sum(adja, axis=1)
    # compute the Laplacian Matrix: L=D-A
    # normailze
    # D^(-1/2) L D^(-1/2)
    L = np.diag(D) - adja
    temp=1.0 / (D ** (0.5))
    temp[temp==inf]=0
    sqrtD = np.diag(temp)
    normL=np.dot(np.dot(sqrtD, L), sqrtD)
    x, V = np.linalg.eig(normL)
    x = x.real.astype(np.float32)
    V = V.real.astype(np.float32)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x:x[0])
    H = np.vstack([V[:,i] for (v, i) in x[:k]]).T
    sc_kmeans = KMeans(n_clusters=k).fit(H)
    
    return sc_kmeans.labels_

def get_Vs(array):
    """Get Vs as defined in the paper by Kleindessner et al (ICML 2019)""" 
    counter = 0
    for i in range(len(array)):
        if array[i] == 1:
            counter += 1
    return counter

def get_N(array): 
    """Helper function to get length of column"""
    return len(array)

def n_ones_vector(array): 
    """Helper function to get an array of ones with length of column"""
    return np.ones(len(array))

def build_matrix_F(G, labels):
    """Build the matrix F as defined in the paper by Kleindessner et al (ICML 2019)"""
    num_samples = len(G) 
    F = []
    for i in range(max(labels)+1):
        column = [0] * len(G)
        for y in range(len(labels)):
            if labels[y] == i:
                column[y] = 1
        column = np.asarray(column - ((get_Vs(column) / get_N(column)) * n_ones_vector(column)))
        F.append(column)       
    return np.transpose((np.asarray(F)))


def unnormalized_fair_spectral(G, n_clusters, groups, random_state):
    """Compute the unnormalized fair spectral clustering on graph G for n_clusters # of clusters"""
    laplacian_matrix = nx.laplacian_matrix(G)
    F = build_matrix_F(G, groups)
    Z = null_space(np.transpose(F))
    LZ = np.matmul(laplacian_matrix.toarray(), Z)
    TZ = np.transpose(Z)
    fed_matrix = np.matmul(TZ, LZ)
    e, v = np.linalg.eigh(fed_matrix)
    v, e = v[:,np.argsort(e)], e[np.argsort(e)]
    km = KMeans(n_clusters=n_clusters)
    H = np.matmul(Z, v[:, :n_clusters])
    H2=v[:, :n_clusters]
    km.fit(H)
    return km.labels_, km.inertia_,H



def unnormalized_spectral(G, n_clusters,random_state):
    laplacian_matrix = nx.laplacian_matrix(G)
    e, v = np.linalg.eigh(laplacian_matrix.toarray())
    v, e = v[:,np.argsort(e)], e[np.argsort(e)]
    H = v[:, :n_clusters]
    km = KMeans(n_clusters=n_clusters)
    
    km.fit(H)
    return km.labels_, km.inertia_