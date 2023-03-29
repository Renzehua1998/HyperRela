'''
存储超边的字典树
'''
# 字典树节点，包括超图节点编号、出现次数和指向孩子的指针（列表）
class Node:
    def __init__(self, val=None, num=0, children=None) -> None:
        self.val = val
        self.num = num
        self.endNum = 0
        self.children = children
class EdgeTrie:
    def __init__(self) -> None:
        self.root = Node()

    ## 在当前位置插入一个孩子节点
    def insert(self, node, cur):
        if not cur.children:  # 当前为叶子节点，直接新建
            newNode = Node(node, 1)
            cur.children = [newNode]
            return newNode
        # 遍历所有子节点，查看是否被插入过
        for ch in cur.children:  
            if node == ch.val:  # 这个节点被插入过
                ch.num += 1
                return ch  # 直接返回
        # 没有被插入过，新建一个
        newNode = Node(node, 1)
        cur.children.append(newNode)
        return newNode

    ## 插入一条超边
    def construct(self, edges):
        # 遍历插入每条边
        for e in edges:
            edge = sorted(e)  # 边中节点保证有序
            cur = self.root  # 从根节点递归插入
            # 一个超图节点是一个字典树节点
            for node in edge:
                cur = self.insert(node, cur)
            cur.endNum += 1  # 作为一条边结束的末尾节点出现的次数
    
    ## 从当前节点的孩子中删除指定超图节点
    def deleteOne(self, node, cur):
        for ch in cur.children:
            if node == ch.val:  # 该节点匹配上，数值-1
                ch.num -= 1
                if ch.num == 0:  # 若数值已经减为0，删除这个节点直接返回
                    cur.children.remove(ch)
                    return None
                return ch
        return None
    
    ## 删除一条超边
    def delete(self, edge):
        edge = sorted(edge)
        cur = self.root  # 从根节点递归删除
        for node in edge:
            cur = self.deleteOne(node, cur)
            if not cur:  # 返回空时说明没找到节点或已经删光，直接返回
                return
        cur.endNum -= 1

    ## 寻找当前位置孩子中是否包含index位标号的节点
    def searchOne(self, s, index, ch):
        if not ch or index >= len(s):  # 到叶子节点或s已经迭代完毕
            return 0
        r = 0
        for i in range(len(ch)):
            if (ch[i].val == s[index]):
                r += ch[i].endNum + self.searchOne(s, index+1, ch[i].children)
            elif ch[i].val < s[index]:
                continue  # 已经计算过了，避免重复计算
            else:
                r += self.searchOne(s, index+1, ch)
        return r
    
    ## 寻找包含于分区s的超边的数目
    def search(self, s):
        s = sorted(s)
        return self.searchOne(s, 0, self.root.children)

if __name__ == '__main__':
    E = [{1, 2}, {1, 2, 4}, {2, 4}, {2, 3, 5, 7}, {1, 4, 6}, {2, 3, 4}]
    edgeTrie = EdgeTrie()
    edgeTrie.construct(E)
    r = edgeTrie.search({1, 2, 3, 4, 6})
    print(r)
    edgeTrie.delete({2, 4})
    r = edgeTrie.search({1, 2, 3, 4, 6})
    print(r)