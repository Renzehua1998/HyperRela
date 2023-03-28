
##########################################################

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

# D:（长度为n列表）节点的度,分别表示每个节点的度数
def Degrees(H,n,m,d):
    M = len(H)
    D = [0]*n
    for i in range(M):
        if(m[i]>0):
            for j in range(n):
                D[j] = D[j] + d[i][j]
    return D

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

##########################################################

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

## 取一个分区和一个边（集）
## 返回新边缘为“活动”的新分区
def newPart(A, s, node_P):
    P = []
    index = 0
    for i in range(len(A)):
        if(len(s.intersection(A[i])) == 0):  # 边和这个分区节点没有交集
            P.append(A[i])  # 原样保留
            for n in A[i]:
                node_P[n] = index
            index += 1
        else:  # 有交集
            s = s.union(A[i])  # 新分区把这个边的所有节点加入进去
    P.append(s)
    for n in s:
        node_P[n] = index
    return P, s

## 取一个分区A，一个被考虑的边s，边集edges，各等级超边数量m，节点度D
## 返回的新分区相比旧分区模块度的增量
def addEdge(A, s, edges, m, D, node_P):
    vol = sum(D)
    M = sum(m)
    # 统计加入新边后改变的分区
    T1 = time.time()
    origA = []
    origSet = set()
    for n in s:
        if node_P[n] not in origSet:
            origSet.add(node_P[n])
            s = s.union(A[node_P[n]])
            origA.append(A[node_P[n]])
    # 度税变化量
    T2 = time.time()
    volA = [0]*len(origA)
    for i in range(len(origA)):
        for j in origA[i]:
            volA[i] = volA[i] + D[j]
        volA[i] = volA[i] / vol
    volP = 0
    for j in s:
        volP += D[j]
    volP = volP / vol
    S = 0
    for i in range(len(m)):
        if (m[i]>0):
            x = (sum([a**i for a in volA]) - volP**i) * m[i] / M
            S = S + x
    T3 = time.time()
    # 边贡献变化量
    r = 0  # 新纳入分区的边数量
    for e in edges:
        if e.issubset(s):
            r += 1
    deltaQ = r / M + S
    T4 = time.time()
    # print('     模块度计算时间:%s微秒' % ((T4 - T1)*1000000))
    wast = [0]*3
    if ((T2 - T1)*1000000 > 0.1): wast[0] = 1
    if ((T3 - T2)*1000000 > 0.1): wast[1] = 1
    if ((T4 - T3)*1000000 > 0.1): wast[2] = 1
    
    wast0 = 0
    if ((T4 - T1)*1000000 > 0.1): wast0 = 1
    return deltaQ, wast, wast0

##########################################################

## 输入超图H，是否print内容
def cnmAlgo(H, verbose=False):
    ## 获取超图H节点的度
    n, m = H_size(H)  # n：节点个数，m：各大小超边的个数列表
    d = d_Degrees(H,n,m)  # len(m)xn矩阵
    D = Degrees(H,n,m,d)  # 长度为n列表
    # 所有超边放到一个列表中
    E = []
    for i in range(len(H)):
        E.extend(H[i])
    ## 初始化返回值
    A_opt = []
    for i in range(n):
        A_opt.extend([{i}])  # 一个节点为一个社区
    q_opt = EdgeContribution(H,A_opt,m) - DegreeTax(A_opt,m,D)
    node_P = [i for i in range(n)]  # 每个节点对应哪个分区
    ## 计算每条边对初始划分的模块度提升
    e_deltaQ = [0]*len(E)
    wast = [[]]*len(E)
    wast0 = [0]*len(E)
    for i in range(len(E)):
        e_deltaQ[i], wast[i], wast0[i] = addEdge(A_opt, E[i], E, m, D, node_P)
    change, tax, contri = 0, 0, 0
    for i in range(len(wast)):
        if (wast[i][0] == 1): change += 1
        if (wast[i][1] == 1): tax += 1
        if (wast[i][2] == 1): contri += 1
    print(change/len(wast), tax/len(wast), contri/len(wast))
    print(wast0.count(1)/len(wast0))
    print(change/wast0.count(1), tax/wast0.count(1), contri/wast0.count(1))
    ## 遍历所有超边，每次放入一个最大模块度提升的超边
    # while len(E)>0: 
    #     e0 = -1
    #     deltaQ_opt = -1
    #     if verbose:  # 输出当前最优模块度和剩余边数
    #         print('best overall:',q_opt, 'edges left: ',len(E))
    #     ## 找出对模块度贡献最大的超边
    #     for i in range(len(E)):
    #         if e_deltaQ[i] > deltaQ_opt:
    #             e0 = i
    #             deltaQ_opt = e_deltaQ[i]
    #     T1 = time.time()
    #     ## 把这个最优超边加进分区内
    #     if(deltaQ_opt > 0):
    #         q_opt = q_opt + deltaQ_opt
    #         A_opt, s_opt = newPart(A_opt, E[e0], node_P)
    #         ## 将这个超边从待选列表中移除
    #         r = []    
    #         for i in range(len(E)):
    #             if(E[i].issubset(s_opt)):
    #                 r.append(i)
    #         for i in range(len(r) - 1, -1, -1):
    #             E.pop(r[i])
    #             e_deltaQ.pop(r[i])
    #         ## 计算更新过的deltaQ
    #         for i in range(len(E)):
    #             if(len(E[i].intersection(s_opt)) != 0):  # 边和这个新分区有交集
    #                 e_deltaQ[i] = addEdge(A_opt, E[i], E, m, D, node_P)
    #         T2 = time.time()
    #         # print('     调整分区时间:%s毫秒' % ((T2 - T1)*1000))
    #     ## 无模块度提升则提前停止
    #     else:
    #         break
    return q_opt, A_opt

if __name__ == '__main__':
    import time
    import pickle
    H, truth, partitions = pickle.load(open('../others/hypergraph-CNM/dblp.pkl','rb'))
    T1 = time.time()
    qC, PC = cnmAlgo(H, verbose=True)
    T2 = time.time()
    print('hyper-CNM运行时间:%s毫秒' % ((T2 - T1)*1000))