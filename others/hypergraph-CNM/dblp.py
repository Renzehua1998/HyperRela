
# coding: utf-8

# # Strict and 2-section Hypergraph Clustering

# In[1]:

import pandas as pd
import numpy as np
import igraph as ig
from sklearn.metrics import adjusted_rand_score as ARI
from strictModularity import *
import time


# In[2]:


def hcut(H,P):
    s = 0
    l = 0
    for i in range(len(H)):
        l = l + len(H[i])
        for j in range(len(P)):
            s = s + sum([x < set(P[j]) for x in H[i] ])
    return (l-s)/l


# In[3]:


import pickle
H, truth, partitions = pickle.load(open('dblp.pkl','rb'))
# PL = partitions[0] ## 2-section Louvain
# PC = partitions[1] ## hypergraph-CNM

# truthDic = []
# truthNew = []
# for i in range(len(truth)):
#     if truth[i] not in truthDic:
#         truthDic.append(truth[i])
#         truthNew.append([])
#     temp = truthDic.index(truth[i])
#     truthNew[temp].append(i)


# In[4]:


## make igraph object
n, m = H_size(H)
e, w = TwoSecEdges(H,m)
g = ig.Graph()
g.add_vertices(n)
g.add_edges([tuple(x) for x in e])
g.es["weight"] = w
g = g.simplify(combine_edges=sum)


# In[5]:


# uncomment to re-compute the Louvain partition
T1 = time.time()
ml = g.community_multilevel(weights="weight")
T2 = time.time()
print('louvain运行时间:%s毫秒' % ((T2 - T1)*1000))
PL = [x for x in ml]


# In[6]:


## Table1 with Louvain
print('Table 1 with Louvain')
print('H-modularity ',modularityH(H,PL))
print('G-modularity ',modularityG(H,PL))
print('hcut ',hcut(H,PL))
print('number of parts ',len(PL))


# In[7]:


# uncomment to re-compute the hyper-CNM partition -- NB: this is slow!
T1 = time.time()
qC, PC = cnmAlgo(H, verbose=True)
T2 = time.time()
print('hyper-CNM运行时间:%s毫秒' % ((T2 - T1)*1000))


# In[8]:


## Table1 with hypergraph-CNM
print('Table 1 with hypergraph-CNM')
print('H-modularity ',modularityH(H,PC))
print('G-modularity ',modularityG(H,PC))
print('hcut ',hcut(H,PC))
print('number of parts ',len(PC))


# In[9]:


def member(P):
    s = sum([len(x) for x in P])
    M = [-1]*s
    for i in range(len(P)):
        for j in list(P[i]):
            M[j]=i
    return M

def edgeLabels(g, gcomm):
    x = [(gcomm[x.tuple[0]]==gcomm[x.tuple[1]]) for x in g.es]
    return x
def AGRI(g, u, v):
    bu = edgeLabels(g, u)
    bv = edgeLabels(g, v)
    su = sum(bu)
    sv = sum(bv)
    suv = sum(np.array(bu)*np.array(bv))
    m = len(bu)
    return((suv - su*sv/m) / (0.5*(su+sv) - su*sv/m))

## cluster similarity
print('Partition similarity measures')
print('ARI ',ARI(member(PL),member(PC)))
print('AGRI ',AGRI(g,member(PL),member(PC)))

# ## cluster results
# print('Partition louvain result measures')
# print('ARI ',ARI(member(truthNew),member(PL)))
# print('AGRI ',AGRI(g,member(truthNew),member(PL)))
# print('Partition hyper-CNM result measures')
# print('ARI ',ARI(member(truthNew),member(PC)))
# print('AGRI ',AGRI(g,member(truthNew),member(PC)))


# In[10]:


def hcutPlus(H,P):
    hc = [-1]*len(H)
    S = 0
    L = 0
    for i in range(len(H)):
        l = len(H[i])
        L = L + l
        s = 0
        for j in range(len(P)):            
            s = s + sum([x < set(P[j]) for x in H[i] ])
        S = S + s
        if l>0:
            hc[i] = (l-s)/l
    return hc


# In[11]:


## Louvain partition
hc = hcutPlus(H,PL)
print('With Louvain partition:')
for i in [2,3,4]:
    print('prop. of edges of size',i,'cut:',hc[i])


# In[12]:


## hyper-CNM partition
hc = hcutPlus(H,PC)
print('With hyper-CNM partition:')
for i in [2,3,4]:
    print('prop. of edges of size',i,'cut:',hc[i])

