'''
@Author:haozhi hcne
@Date:2021-12-14
@Target:netowrk instance for network construction and reserve

这里我们构建一个网络实例，其作用是
1、构建网络，并存储
2、其中的方法和函数用于网络拓扑等结构的计算


补充：新增加了剔除权重=0的边的网络图12-28进行了修复
'''
import time
import numpy as np
import networkx as nx
import networkx.algorithms as nxa
import networkx.algorithms.community as  nxac
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt

class networkinstance():

    def __init__(self,nodes,edges,weightlist,date,G):
        self.Nodes = nodes # list
        self.Edges = edges # list
        self.Weightlist = weightlist  #包含weight的edgeslist:（i,j,{weight:4.2}）
        self.Date = date
        self.G = G

    #print(实例）直接打印返回值
    def __str__(self):
        return f'listofnodes {self.Nodes}'
    #输出全部网络实例的数据，随时支持调用
    def getDate(self):
        return self.Date

    def getNodes(self):
        return self.Nodes

    def getNetwork(self):
         return self.G

    def setNetwork(self,newg):
        self.G = newg

    def getWeightlist(self):
        return self.Weightlist

    def plotGraph(self):
        # networkx 提供多种布局方式，有draw_circular,draw_spring等。需要根据需求参考文档
        nx.draw(self.G,with_labels=True,font_weight='bold')
        plt.show()

    # return the elements of graph includes: nodes edges adj (节点，边，相邻）
    def return_element(self):
        list_nodes = list(self.G.nodes)
        list_edges = list(self.G.edges)
        list_adjcent = list(self.G.adj)
        return list_nodes,list_edges,list_adjcent

# 生成一个剔除了权重=0的边的网络，并且返回一个新的网络实例
    ''':return
        Network: object
    '''
    def cut_zeroweight_edges(self):
        newgraph = nx.Graph()
        # 剔除权重为0的边，这样就形成了一个稀疏图。之前没有提出权重为0的边还是有一些问题的。
        # 同时，我们也保持了权重为0的边
        edges_weigth = [(x, y, {'weight': w}) for x, y, w in self.G.edges.data('weight') if w > 0]
        newgraph.add_edges_from(edges_weigth)

        return networkinstance(self.Nodes, self.Edges, self.Weightlist, self.Date, newgraph)

# 一个生成剔除一定阈值比的网络，我们测试一下即可，并不一定使用
    ''':return
        network:object
    根据阈值来筛选图结构，形成阈值图
    '''
    def compute_Threshold(self, threshold):
        thresholdgraph = nx.Graph()
        # 剔除一定阈值的边
        edges_weight = [(x, y, {'weight': v}) for x, y, v in self.G.edges.data('weight') if v > threshold]
        thresholdgraph.add_edges_from(edges_weight)

        return networkinstance(self.Nodes, self.Edges, self.Weightlist, self.Date, thresholdgraph)

    #数生成的算法全部是重新生成一个network的instance，因此，需要注意返回的对象是一个instance，其中的变化的G的部分
    #图生成算法部分的实现！！
    ''':return
        Network : object
    返回的是一个Network的实例，即通过生成网络创建一个新的网络并存储到Network类中
    '''
    def compute_PMFG(self):

        def sort_graph_edges(G):
            sorted_edges = []
            for source, dest, data in sorted(G.edges(data=True), key=lambda x: x[2]['weight'],
                                             reverse=True):  # in descending order!
                sorted_edges.append({'source': source,
                                     'dest': dest,
                                     'weight': data['weight']})
            return sorted_edges

        PMFG = nx.Graph()  # initialize
        ne_total = self.G.number_of_edges()
        nb_nodes = len(self.G.nodes)
        ne_pmfg = 3 * (nb_nodes - 2)
        sorted_edges = sort_graph_edges(self.G)
        t0 = time.time()
        for i, edge in enumerate(sorted_edges):
            PMFG.add_edge(edge['source'], edge['dest'], weight=edge['weight'])
            if not nx.algorithms.planarity.check_planarity(PMFG)[0]:
                PMFG.remove_edge(edge['source'], edge['dest'])
            ne = PMFG.number_of_edges()
            print(
                "Generating PMFG... added edges in PMFG %d/%d (%.2f%%) lookup edges in G %d/%d (%.2f%%) Elapsed TIme %.2f [sec]" \
                % (ne, ne_pmfg, (ne / ne_pmfg) * 100, i, ne_total, (i + 1 / ne_total) * 100, time.time() - t0),
                end="\r")
            if ne == ne_pmfg:
                break

        return networkinstance(self.Nodes, self.Edges, self.Weightlist, self.Date, PMFG)


    #对于每一个对象，重新生成一个最小生成树的网络对象，其中包括的内容是这个网络结构的全部，只不过是最小生成树的结果
    ''':param
        algorithm:str ('prim','kruskal')
    两种主要的MST的生成算法，这里需要手动选择，否则默认为：prim
    '''
    ''':return
            Network : object
    返回的是一个Network的实例，即通过生成网络创建一个新的网络并存储到Network类中
    '''
    def compute_MST(self,algorithms):

        print('minimum spanning tree generation in processing----------------------------------------')
        if algorithms == 'prim':
            MSTG = nx.minimum_spanning_tree(self.G, weight='weight', algorithm='prim', ignore_nan=True)
        else:
            MSTG = nx.minimum_spanning_tree(self.G, weight='weight', algorithm='kruskal', ignore_nan=True)

        return networkinstance(self.Nodes, self.Edges, self.Weightlist, self.Date, MSTG)




