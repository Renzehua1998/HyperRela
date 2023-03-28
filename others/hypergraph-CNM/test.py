import pandas as pd
import numpy as np
import igraph as ig
from sklearn.metrics import adjusted_rand_score as ARI
from strictModularity import *
import time

import pickle
H, truth, partitions = pickle.load(open('dblp.pkl','rb'))
# PL = partitions[0] ## 2-section Louvain
# PC = partitions[1] ## hypergraph-CNM

## make igraph object
n, m = H_size(H)
e, w = TwoSecEdges(H,m)
g = ig.Graph()
g.add_vertices(n)
g.add_edges([tuple(x) for x in e])
g.es["weight"] = w
g = g.simplify(combine_edges=sum)

# # uncomment to re-compute the Louvain partition
# T1 = time.time()
# ml = g.community_multilevel(weights="weight")
# T2 = time.time()
# print('louvain运行时间:%s毫秒' % ((T2 - T1)*1000))
# PL = [x for x in ml]
# print(PL)

# uncomment to re-compute the hyper-CNM partition -- NB: this is slow!
T1 = time.time()
qC, PC = cnmAlgo(H, verbose=True)
T2 = time.time()
print('hyper-CNM运行时间:%s毫秒' % ((T2 - T1)*1000))