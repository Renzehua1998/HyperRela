> HyperNetX
>
> 仓库地址：https://github.com/pnnl/HyperNetX
>
> 在线文档：https://pnnl.github.io/HyperNetX

HNX库提供了用于将复杂网络中的实体和关系建模为超图的类和方法

这个库是我们在探索超图能告诉我们什么时发现最有用的方法和算法的存储库

# 概述

HyperNetX（HNX）库的开发是为了支持研究人员将数据建模为超图

- 1.0版超图构造函数可以读取Pandas数据帧对象，并基于列标题创建边和节点
- C++插件NWHy可以在Linux环境中使用，以支持优化的超图方法，如s-中心性度量
- 1.1版用于基于超边缘关联和加权对顶点进行聚类的聚类模块
- 用于合成ChungLu和DCSBM超图的生成器模块
- 1.2版中的新功能：增加了**模块化和聚类的算法模块和教程**（重点学习）

# 安装

```shell
conda create -n <env name> python=3.7
source activate <env name>
pip install hypernetx
```

主要是`pip install hypernetx`

- 可以使用[NWHy](https://github.com/pnnl/NWHypergraph)框架进行并行加速。如果您需要使用NWHy，则需要在环境中使用python=3.9版本和conda版本的tbb库

# 术语表

1. **二分条件**（Bipartite Condition）：

   对类EntitySet的实例施加的条件。作为同一EntitySet的元素的实体不能相互包含为元素\*。实体集的元素和孩子为二分图生成特定的分区。分区同构于Hypergraph，其中元素对应于超边，子元素对应于节点。EntitySet是用于构造动态超图NX的基本对象。

2. **度**（degree）：

   给定超图（节点，边），节点中节点的度是节点所属边的数量。

3. **对偶**（dual）：

   对于超图（节点、边），其对偶是通过切换节点和边的角色而构建的超图。更准确地说，如果节点i属于超图中的边j，那么节点j属于对偶超图中边i。

4.  **实体**（Entity）：

   entity.py中的类，节点、边和其他HNX结构的基类。一个实体有一个唯一的id、一组财产和一组其他属于它的实体，称为它的元素（一个实体可能不包含它自己）。如果一个实体A属于另一个实体B，那么A在B中具有成员资格，而A是B的元素。对于任何实体A，使用A.elements访问其元素的字典（由uid键控），使用A.membership访问其成员资格的字典。

   - Entity.children：属性——返回与本实体具有从属关系的实体
   - Entity.depth：方法——返回一个实体的非空孩子集的数量
   - Entity.elements：属性——返回实体元素的字典
   - Entity.levelset：方法——多层级孩子集合
   - Entity.memberships：属性——实体所属实体的uid、实体键值对的字典。
   - Entity.registry：属性——一个实体的子级的uid、实体键值对的字典

5. **实体集**（entityset）：

   满足二部分条件的实体A，即A的级别1中的实体集与A的级别2中的实体集中不相交的属性，即A中的元素与A的子级不相交。实体集在类entityset中实例化。

……待续

# 详细API文档

1. [Hypergraphs](https://pnnl.github.io/HyperNetX/build/classes/modules.html)
2. [Algorithms](https://pnnl.github.io/HyperNetX/build/algorithms/modules.html)
3. [Drawing](https://pnnl.github.io/HyperNetX/build/drawing/modules.html)
4. [Reports](https://pnnl.github.io/HyperNetX/build/reports/modules.html)

# C++优化插件NWHy

提供了许多超图方法的优化C++实现。NWHy是一个可扩展的高性能超图库。它有三个依赖项。

1. NWGraph库：提供图数据结构、一组丰富的图形数据结构适配器以及各种高性能图算法实现。
2. 英特尔OneAPI线程构建块（oneTBB）：提供并行性
3. Pybind11：将NWHy封装为一个python模块

至少需要python3.9环境

# 绘图插件 HyperNetXWidget

> 仓库地址：https://github.com/pnnl/hypernetx-widget
>
> 在线演示：https://pnnl.github.io/hypernetx-widget/

将HNX的内置可视化功能扩展到基于JavaScript的交互式可视化。该工具有两个主要界面，超图可视化和节点和边面板。

# 模块度和聚类（重点）

HNX中的超图模块化子模块提供了为超图中给定的顶点划分计算超图模块性的函数。通常，较高的模块性表示将顶点更好地划分为密集社区。

提供了两个生成此类超图分区的函数：**Kumar算法**和简单的**Last Step精化算法**。

子模块还提供了一个为给定超图生成两部分图（普通图）的函数，然后可以使用该函数通过基于图的算法找到顶点分区。