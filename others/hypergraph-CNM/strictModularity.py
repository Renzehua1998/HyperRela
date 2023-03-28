##########################################################
## Hypergraph algorithms 
## Quick Python implementation - small datasets only!
##
## Data representation
##
## h: list of sets (the hyperedges), with 0-based integer vertices
##    example: h = [{0,1},{1,2,3},{2,3,4},{4,6},{5,6}]
## H = list2H(h) (hypergraph)
##    example: H = [[], [], [{0, 1}, {4, 6}, {5, 6}], [{1, 2, 3}, {2, 3, 4}]]
## 
## qH, PH = randomAlgo(H, steps=10, verbose=True, ddeg=False)
## qH, PH = cnmAlgo(H, verbose=True)
## qG, PG = randomAlgoTwoSec(H, steps=10, verbose=True)
## 
##########################################################

from scipy.special import comb as choose
import itertools
from random import shuffle, randint, sample
import time

##########################################################

## return: number of nodes (recall: 0-based), and
## number of edges of each cardinality
#  统计超图节点个数和各等级超边个数（列表）
#  n：节点个数，m：各大小超边的个数列表
def H_size(H):
    M = len(H)
    m = []
    n = 0
    for i in range(M):
        m.append(len(H[i]))
        if(len(H[i])>0):
            j = max(set.union(*H[i]))
            if j>n:
                n = j
    return n+1, m

## vertex d-degrees for each d
# d:len(m)×n矩阵，每一行都是一个等级超边中节点的统计，
# 其中n个数分别表示每个节点在此规模超边中出现的次数
def d_Degrees(H, n, m):
    M = len(H)
    d = [[]]*M
    for i in range(M):
        if(m[i]>0):
            x = [y for x in H[i] for y in list(x)]
            y = [x.count(i) for i in range(n)]
            d[i] = y
    return d

## vertex total degrees
# D:（长度为n列表）节点的度,分别表示每个节点的度数
def Degrees(H,n,m,d):
    M = len(H)
    D = [0]*n
    for i in range(M):
        if(m[i]>0):
            for j in range(n):
                D[j] = D[j] + d[i][j]
    return D

##########################################################

## edge contribution: given (H,A,m)
# ec:边贡献（全部包含着在社区内的边占比）
def EdgeContribution(H,A,m):
    ec = 0
    for i in range(len(H)):
        for j in range(len(H[i])):
            for k in range(len(A)):
                if(H[i][j].issubset(A[k])):
                    ec = ec + 1
                    break
    ec = ec / sum(m)
    return ec

##########################################################

## degree tax - with d-degrees as null model
# d-等级超边节点度作为空模型的度税
def d_DegreeTax(A,m,d):
    dt = 0
    for i in range(len(m)):  # 遍历所有等级的超边，把结果累加
        if (m[i]>0):
            S = 0
            for j in range(len(A)):  # 遍历所有分区，把分区中每个节点在此等级超边下的度累加，取i次方
                s = 0
                for k in A[j]:
                    s = s + d[i][k]
                s = s ** i
                S = S + s  # 再把所有分区的结果加起来
            S = S / (i**i * m[i]**(i-1) * sum(m))  # 归一化
            dt = dt + S  # 累加到总结果中
    return dt
    
## degree tax - with degrees as null model
# 整体节点度作为空模型的度税（通用度税）
def DegreeTax(A,m,D):
    dt = 0
    vol = sum(D)
    M = sum(m)
    ## vol(A_i)'s
    volA = [0]*len(A)
    for i in range(len(A)):
        for j in A[i]:
            volA[i] = volA[i] + D[j]
        volA[i] = volA[i] / vol
    ## sum over d
    S = 0
    for i in range(len(m)):
        if (m[i]>0):
            x = sum([a**i for a in volA]) * m[i] / M
            S = S + x
    return S

##########################################################

## 2-section: return extended list of edges and edge weights 
# 把超图扩展为带权普通图（权重均分到完全图每条边）
def TwoSecEdges(H,m):
    e = []
    w = []
    for i in range(len(m)):
        if(m[i]>0 and i>1):
            den = choose(i,2)
            for j in range(len(H[i])):
                s = [set(k) for k in itertools.combinations(H[i][j],2)]
                x = [1/den]*len(s)
                e.extend(s)
                w.extend(x)
    return e,w 

# 二元图的边贡献
def TwoSecEdgeContribution(A,e,w):
    ec = 0
    for i in range(len(A)):
        for j in range(len(e)):
            if(e[j].issubset(A[i])):
                ec = ec + w[j]
    ec = ec / sum(w)
    return ec

# 二元图的度
def TwoSecDegrees(n,e,w):
    d = [0]*n
    for i in range(len(e)):
        d[list(e[i])[0]] = d[list(e[i])[0]] + w[i]
        d[list(e[i])[1]] = d[list(e[i])[1]] + w[i]
    return d

# 二元图的度税
def TwoSecDegreeTax(A,d):
    dt = 0
    for i in range(len(A)):
        s = 0
        for j in list(A[i]):
            s = s + d[j]
        s = s**2
        dt = dt + s
    dt = dt / (4*(sum(d)/2)**2)
    return dt

##########################################################

## take a partition and an edge (set)
## return new partition with new edge "active"
## 取一个分区和一个边（集）
## 返回新边缘为“活动”的新分区
def newPart(A,s):
    P = []
    for i in range(len(A)):
        if(len(s.intersection(A[i])) == 0):  # 边和这个分区节点没有交集
            P.append(A[i])  # 什么也不做
        else:  # 有交集
            s = s.union(A[i])  # 新分区把这个边的所有节点加入进去
    P.append(s)
    return P

##########################################################

## 输入超图H，是否print内容，是否使用d-等级超边模块度
def cnmAlgo(H, verbose=False, ddeg=False):
    ## get degrees from H  获取超图节点的度
    n, m = H_size(H)  # n：节点个数，m：各大小超边的个数列表
    d = d_Degrees(H,n,m)  # len(m)xn矩阵
    D = Degrees(H,n,m,d)  # 长度为n列表
    ## get all edges in a list
    e = []  # 所有超边放到一个列表中
    for i in range(len(H)):
        e.extend(H[i])
    ## initialize modularity, partition
    ## 初始化返回值
    A_opt = []
    for i in range(n):
        A_opt.extend([{i}])  # 一个节点为一个社区
    if ddeg:
        q_opt = EdgeContribution(H,A_opt,m) - d_DegreeTax(A_opt,m,d)  
    else:
        q_opt = EdgeContribution(H,A_opt,m) - DegreeTax(A_opt,m,D)
    ## e contains the edges NOT yet in a part
    while len(e)>0:  # 遍历所有超边，每次放入一个最大模块度提升的超边
        q0 = -1
        e0 = -1
        if verbose:  # 输出当前最优模块度和剩余边数
            print('best overall:',q_opt, 'edges left: ',len(e))
        ## pick best edge to add .. this is slow as is!
        ## 找出对模块度贡献最大的超边
        for i in range(len(e)):
            P = newPart(A_opt,e[i])
            if ddeg:
                q = EdgeContribution(H,P,m) - d_DegreeTax(P,m,d)
            else:
                q = EdgeContribution(H,P,m) - DegreeTax(P,m,D)
            if q>q0:
                e0 = i
                q0 = q
        ## add best edge found if any
        ## 把这个最优超边加进分区内
        if(q0 > q_opt):
            q_opt = q0
            A_opt = newPart(A_opt,e[e0])
            ## remove all 'active' edges
            ## 将这个超边从待选列表中移除
            r = []    
            for i in range(len(e)):
                for j in range(len(A_opt)):
                        if(e[i].issubset(A_opt[j])):
                            r.append(e[i])
                            break
            for i in range(len(r)):
                e.remove(r[i])
        ## early stop if no immediate improvement
        else:
            break
    return q_opt, A_opt

##########################################################

## random algorithm - start from singletons, add edges w.r.t. permutation IF q improves
## 随机扩展普通图算法，指定迭代次数，每次添加一个超边，保留普通模块度最大的，贪婪迭代
def randomAlgoTwoSec(H, steps=10, verbose=False):
    ## get degrees from H
    n, m = H_size(H) 
    ed, w = TwoSecEdges(H, m)  # 扩展为普通图，返回边列表和权值列表
    d = TwoSecDegrees(n, ed, w)
    ## get all edges in H
    e = []
    for i in range(len(H)):
        e.extend(H[i])
    ## initialize modularity, partition
    q_opt = -1
    A_opt= []
    ## algorithm - go through random permutations
    for ctr in range(steps):
        ## Loop here
        shuffle(e)
        ## list of singletons
        A = []
        for i in range(n):
            A.extend([{i}])
        ## starting (degree) modularity
        q0 = TwoSecEdgeContribution(A, ed, w) - TwoSecDegreeTax(A, d)
        for i in range(len(e)):
            P = newPart(A,e[i])
            q = TwoSecEdgeContribution(P,ed, w) - TwoSecDegreeTax(P, d)
            if q > q0:
                A = P
                q0 = q
        if q0 > q_opt:
            q_opt = q0
            A_opt = A
        if verbose:
            print('step',ctr,':',q_opt)
    return q_opt, A_opt


## random algorithm - start from singletons, add edges w.r.t. permutation IF q improves
## 随机超图算法，指定迭代次数，每次添加一个超边，保留超图模块度最大的，贪婪迭代
def randomAlgo(H, steps=10, verbose=False, ddeg=False):
    ## get degrees from H
    n, m = H_size(H) 
    d = d_Degrees(H,n,m)
    D = Degrees(H,n,m,d)
    ## get all edges in H
    e = []
    for i in range(len(H)):
        e.extend(H[i])
    ## initialize modularity, partition
    q_opt = -1
    A_opt= []
    ## algorithm - go through random permutations
    for ctr in range(steps):
        ## Loop here
        shuffle(e)
        ## list of singletons
        A = []
        for i in range(n):
            A.extend([{i}])
        ## starting (degree) modularity
        if ddeg:
            q0 = EdgeContribution(H,A,m) - d_DegreeTax(A,m,d)
        else:   
            q0 = EdgeContribution(H,A,m) - DegreeTax(A,m,D)
        for i in range(len(e)):
            P = newPart(A,e[i])
            if ddeg:
                q = EdgeContribution(H,P,m) - d_DegreeTax(P,m,d)
            else:
                q = EdgeContribution(H,P,m) - DegreeTax(P,m,D)
            if q > q0:
                A = P
                q0 = q
        if q0 > q_opt:
            q_opt = q0
            A_opt = A
        if verbose:
            print('step',ctr,':',q_opt)
    return q_opt, A_opt

##########################################################

## Map vertices 0 .. n-1 to their respective 0-based part number
## 返回每个节点对应的分区编号
def PartitionLabels(P):
    n = 0
    for i in range(len(P)):
        n = n + len(P[i])
    label = [-1]*n
    for i in range(len(P)):
        l = list(P[i])
        for j in range(len(l)):
            label[l[j]] = i
    return label

##########################################################

## generate m edges between [idx1,idx2] inclusively
## of size between [size1,size2] inclusively
## Store in a list of lists of sets
##在[idx1，idx2]之间生成m条边（包括）
##尺寸介于[size1，size2]之间（含）
##存储在集合列表中
def generateEdges(m,idx1,idx2,size1,size2):
    ## init
    L = [[]]*(size2+1)
    for i in range(size2+1):
        L[i]=[]
    v = list(range(idx1,idx2+1))
    if size2>len(v):
        size2 = len(v)
    ## generate - never mind repeats for now
    for i in range(m):
        size = randint(size1,size2)
        L[size].append(set(sample(v,size)))
    return L  

## merge two lists of lists of sets
## 合并两个集合列表 
def mergeEdges(L1,L2):
    l = max(len(L1),len(L2))
    L = [[]]*l
    for i in range(len(L1)):
        L[i] = L1[i]
    for i in range(len(L2)):
        L[i] = L[i] + L2[i]
    ## uniquify
    for i in range(l):
        L[i] = [set(j) for j in set(frozenset(i) for i in L[i])]
    return L

##########################################################

## format Hypergraph given list of hyperedges (list of sets of 0-based integers)
## 把原始超边列表转换为分层次的超边集
def list2H(h):
    ml = max([len(x) for x in h])
    H = [[]]*(ml+1)
    for i in range(ml+1):
        H[i] = []
    for i in range(len(h)):
        l = len(h[i])
        H[l].append(h[i])
    return H

## two section modularity
## 普通图模块度
def modularityG(H,A):
    n, m = H_size(H) 
    ed, w = TwoSecEdges(H, m)
    d = TwoSecDegrees(n, ed, w)
    return(TwoSecEdgeContribution(A, ed, w) - TwoSecDegreeTax(A, d))

## strict H-modularity
## 严格超图模块度
def modularityH(H,A,ddeg=False):
    n, m = H_size(H) 
    d = d_Degrees(H,n,m)
    D = Degrees(H,n,m,d)
    if ddeg:
        return(EdgeContribution(H,A,m) - d_DegreeTax(A,m,d))
    else:
        return(EdgeContribution(H,A,m) - DegreeTax(A,m,D))

##########################################################
