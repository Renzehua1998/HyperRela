
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
def newPart(A,s):
    P = []
    for i in range(len(A)):
        if(len(s.intersection(A[i])) == 0):  # 边和这个分区节点没有交集
            P.append(A[i])  # 什么也不做
        else:  # 有交集
            s = s.union(A[i])  # 新分区把这个边的所有节点加入进去
    P.append(s)
    return P

## 取一个分区A和一个边（集）s
## 返回的新分区相比旧分区模块度的增量
def addEdge(A, s, edges, m, D):
    vol = sum(D)
    M = sum(m)
    # 统计加入新边后改变的分区
    origA = []
    for i in range(len(A)):
        if(len(s.intersection(A[i])) != 0):  # 边和这个分区节点有交集
            s = s.union(A[i])  # 新分区把这个边的所有节点加入进去
            origA.append(A[i])
    # 度税变化量
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
    # 边贡献变化量
    r = 0  # 新纳入分区的边数量
    for e in edges:
        if e.issubset(s):
            r += 1
    deltaQ = r / M + S
    return deltaQ, s

##########################################################

## 输入超图H，是否print内容
def cnmAlgo(H, verbose=False):
    ## 获取超图H节点的度
    n, m = H_size(H)  # n：节点个数，m：各大小超边的个数列表
    d = d_Degrees(H,n,m)  # len(m)xn矩阵
    D = Degrees(H,n,m,d)  # 长度为n列表
    # 所有超边放到一个列表中
    e = []
    for i in range(len(H)):
        e.extend(H[i])
    ## 初始化返回值
    A_opt = []
    for i in range(n):
        A_opt.extend([{i}])  # 一个节点为一个社区
    q_opt = EdgeContribution(H,A_opt,m) - DegreeTax(A_opt,m,D)
    ## 遍历所有超边，每次放入一个最大模块度提升的超边
    while len(e)>0: 
        e0 = -1
        deltaQ_opt = -1
        s_opt = []
        if verbose:  # 输出当前最优模块度和剩余边数
            print('best overall:',q_opt, 'edges left: ',len(e))
        ## 找出对模块度贡献最大的超边
        T1 = time.time()
        for i in range(len(e)):
            # P = newPart(A_opt,e[i])
            # q = EdgeContribution(H,P,m) - DegreeTax(P,m,D)
            T4 = time.time()
            deltaQ, s = addEdge(A_opt, e[i], e, m, D)
            if deltaQ > deltaQ_opt:
                e0 = i
                deltaQ_opt = deltaQ
                s_opt = s
            T5 = time.time()
            # print('         计算模块度变化量时间:%s毫秒' % ((T5 - T4)*1000))
        T2 = time.time()
        # print('     找最优超边时间:%s毫秒' % ((T2 - T1)*1000))
        ## 把这个最优超边加进分区内
        if(deltaQ_opt > 0):
            q_opt = q_opt + deltaQ_opt
            A_opt = newPart(A_opt,e[e0])
            ## 将这个超边从待选列表中移除
            r = []    
            for i in range(len(e)):
                if(e[i].issubset(s_opt)):
                    r.append(e[i])
            for i in range(len(r)):
                e.remove(r[i])
            T3 = time.time()
            # print('     调整分区时间:%s毫秒' % ((T3 - T2)*1000))
        ## 无模块度提升则提前停止
        else:
            break
    return q_opt, A_opt

if __name__ == '__main__':
    import time
    import pickle
    H, truth, partitions = pickle.load(open('dblp.pkl','rb'))
    T1 = time.time()
    qC, PC = cnmAlgo(H, verbose=True)
    T2 = time.time()
    print('hyper-CNM运行时间:%s毫秒' % ((T2 - T1)*1000))
