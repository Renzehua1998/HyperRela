
# coding: utf-8

# # Strict and 2-section Hypergraph Clustering

# In[1]:

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as Rand
import igraph as ig
from scipy import stats

## All hypergraph functions are here
from strictModularity import *

np.random.seed(0)
import random
random.seed(0)
# In[2]:


## proportion of hyperedges cuted w.r.t. partition P
def hcut(H,P):
    s = 0
    l = 0
    for i in range(len(H)):
        l = l + len(H[i])
        for j in range(len(P)):
            s = s + sum([x < set(P[j]) for x in H[i] ])
    return (l-s)/l


# In[3]:


## number of hyperedge of each type
pointsPerLine = 30
numOutliers = 60
REP = 100
nl = [0]*pointsPerLine + [1]*pointsPerLine + [2]*pointsPerLine + [3]*numOutliers

## is this edge from the same line?
def lineEdge(e):
    s = set([nl[i] for i in e])
    return (len(s)==1 and 3 not in s)

L = []
mu = 0.33 ## proportion of noisy edges to keep

for z in [0.5,2,1]: ## ratios of 3-edges and 4-edges

    fn = './hypergraph_3uniform.csv'
    x3 = np.loadtxt(fn, delimiter=',', dtype='int')
    fn = './hypergraph_4uniform.csv'
    x4 = np.loadtxt(fn, delimiter=',', dtype='int')
    ## Downsample
    p3 = .1 * z
    base3 = [set(i) for i in x3 if np.random.random()<p3]
    p4 = .035 / z
    base4 = [set(i) for i in x4 if np.random.random()<p4]
    base = base3+base4
    
    for rep in range(REP):
        ## sample down False cases to get mu
        x = [lineEdge(i) for i in base]
        t = x.count(True)
        f = x.count(False)
        p = mu*t/((1-mu)*f)
        h = [base[i] for i in range(len(base)) if x[i] or np.random.sample()<p]
        H = list2H(h)
        ## 2-section edges and weights
        n, m = H_size(H)
        e, w = TwoSecEdges(H,m)
        ## make weighted igraph object
        g = ig.Graph()
        g.add_vertices(n)
        g.add_edges([tuple(x) for x in e])
        g.es["weight"] = w
        g = g.simplify(combine_edges=sum)
        ## Louvain partition
        ml = g.community_multilevel(weights="weight")
        P = [x for x in ml]
        l = [len(x) for x in H]
        ## modularities and hcut
        L.append([l[3]/(l[3]+l[4]),modularityH(H,P),modularityG(H,P),hcut(H,P)])
      


# In[5]:


D = pd.DataFrame(L, columns=['ratio_3','qH','qG','hcut'])
x = D['ratio_3']>.6
y = D['ratio_3']>.4
D['regime'] = [int(x[i])+int(y[i]) for i in range(len(x))]


# In[9]:


X = D[D['regime']==0]
plt.plot(X['hcut'],X['qG'],'*r',label='majority of 4-edges')
X = D[D['regime']==2]
plt.plot(X['hcut'],X['qG'],'*b',label='majority of 3-edges')
X = D[D['regime']==1]
plt.plot(X['hcut'],X['qG'],'*g',label='balanced')
plt.legend();
plt.xlabel('Hcut value',fontsize=14)
plt.ylabel('Graph modularity', fontsize=14)
slope, intercept, r, p, stderr = stats.linregress(X['hcut'],X['qG'])
print('Balanced case -- Slope:', slope,' R_squared:',r*r)
plt.savefig('lines_qG.png')


# In[10]:

plt.figure()
X = D[D['regime']==0]
plt.plot(X['hcut'],X['qH'],'*r',label='majority of 4-edges')
X = D[D['regime']==2]
plt.plot(X['hcut'],X['qH'],'*b',label='majority of 3-edges')
X = D[D['regime']==1]
plt.plot(X['hcut'],X['qH'],'*g',label='balanced')
plt.legend()
plt.xlabel('Hcut value',fontsize=14)
plt.ylabel('Hypergraph modularity', fontsize=14)
slope, intercept, r, p, stderr = stats.linregress(X['hcut'],X['qH'])
print('Balanced case -- Slope:', slope,' R_squared:',r*r)
plt.savefig('lines_qH.png')

