'''
@Author: haozhi chen
@Date：2022-09
@Target：实现topological index的计算，基本算法均来自于NetworkX库

考虑方案1：
（1）结合中心性，连通性算法

考虑方案2：
（1）只是用中心性算法进行研究 #有相关研究进行波动率溢出的预测
2023-02 悄悄测试一下
'''

import networkx as nx
import pandas as pd
import numpy as np

import threading # 多线程方案
import time


'''
根据我们的研究设定，我们在这里提取一些指标出来
totcal connectedness：该值需要考虑权重，那么就有in out total，这三个部分的权重
connectivity：根据普分解得到的结果
closeness centrality：接近中心性
betweenness centrality：中介中心性

'''
def topological_index(G):
    print('--------------------Total connectedness in processing----------------------')
    nodes = list(G.nodes)
    nodes.sort() # 所有节点，并且排序
    edges = G.edges.items() # item()返回的是 e,dict 其中 e=(e1,e2)的边，dict={‘weight’：num}权重或者是自命名的东西
    "这里的connectedness需要考虑一个有向图的问题，即出，入，全部的情况"
    in_tc,out_tc,tc = {},{},{}
    for node in nodes:
        value_in,value_out,value_tc = 0,0,0
        for e,weight_dict in edges:
            if node in e: # 如果节点再边中
                value_tc = value_tc + weight_dict.get("weight") # 获取全部权重值
            if node in [e[0]]:
                value_out = value_out + weight_dict.get("weight") # 获取出度权重
            if node in [e[1]]:
                value_in = value_in + weight_dict.get("weight") # 获取入度权重
            else:
                pass
        in_tc[node] = value_in # 向外传导的权重
        out_tc[node] = value_out # 向内传导的权重
        tc[node] = value_tc # 综合整体单个节点的全部权重！
    TC = tc


    print('--------------------Connectivity in processing------------------------------')
    ANC = nx.all_pairs_node_connectivity(G) # 这是一个flow-based的计算，统计的是必须移除的最小节点个数
    anc_dict = {}
    for k,v in ANC.items():
        meanv = np.mean(list(v.values()))
        anc_dict[k] = meanv # 因为ANC中是一个节点 -对- 多个节点的情况。需要对多个节点的结果进行求均值。方可采用！
    ANC = anc_dict

    print('--------------------Closeness centrality in processing----------------------')
    CC = nx.closeness_centrality(G,distance='weight')

    print('--------------------Betweenness centrality in processing--------------------')
    BEC = nx.betweenness_centrality(G,weight='weight')



    return TC,ANC,CC,BEC

''':进行多进程工作的布置
1）将不同的指标计算 ——> 纳入到不同的函数中
2）多进程调用这些函数
'''
"Total connectedness"
def tc_func(G):
    print('--------------------Total connectedness in processing----------------------')
    start_time = time.time()
    nodes = list(G.nodes)
    nodes.sort() # 所有节点，并且排序
    edges = G.edges.items() # item()返回的是 e,dict 其中 e=(e1,e2)的边，dict={‘weight’：num}权重或者是自命名的东西
    "这里的connectedness需要考虑一个有向图的问题，即出，入，全部的情况"
    in_tc,out_tc,tc = {},{},{}
    for node in nodes:
        value_in,value_out,value_tc = 0,0,0
        for e,weight_dict in edges:
            if node in e: # 如果节点再边中
                value_tc = value_tc + weight_dict.get("weight") # 获取全部权重值
            if node in [e[0]]:
                value_out = value_out + weight_dict.get("weight") # 获取出度权重
            if node in [e[1]]:
                value_in = value_in + weight_dict.get("weight") # 获取入度权重
            else:
                pass
        in_tc[node] = value_in # 向外传导的权重
        out_tc[node] = value_out # 向内传导的权重
        tc[node] = value_tc # 综合整体单个节点的全部权重！
    TC = tc
    end_time = time.time()
    print(f'-----------Total connectedness running time is {end_time - start_time}---------')
    return TC

def anc_func(G):
    print('--------------------Connectivity in processing------------------------------')
    start_time = time.time()
    ANC = nx.all_pairs_node_connectivity(G) # 这是一个flow-based的计算，统计的是必须移除的最小节点个数
    anc_dict = {}
    for k,v in ANC.items():
        meanv = np.mean(list(v.values()))
        anc_dict[k] = meanv # 因为ANC中是一个节点 -对- 多个节点的情况。需要对多个节点的结果进行求均值。方可采用！
    ANC = anc_dict
    end_time = time.time()
    print(f'-----------Connectivity running time is {end_time - start_time}---------')
    return ANC

#####################################################################################
"中心性算法"
def cc_func(G):
    print('--------------------Closeness centrality in processing----------------------')
    start_time = time.time()
    CC = nx.closeness_centrality(G,distance='weight')
    end_time = time.time()
    print(f'-----------Closeness centrality running time is {end_time - start_time}---------')
    return CC

def bec_func(G):
    print('--------------------Betweenness centrality in processing--------------------')
    start_time = time.time()
    BEC = nx.betweenness_centrality(G,weight='weight')
    end_time = time.time()
    print(f'-----------Betweenness centrality running time is {end_time - start_time}---------')
    return BEC

def dc_func(G):
    print('--------------------Degree centrality in processing--------------------------')
    start_time = time.time()
    DC = nx.degree_centrality(G)
    end_time = time.time()
    print(f'-----------Degree centrality running time is {end_time - start_time}---------')
    return DC

'pageRank 算法'
def pr_func(G):
    print('--------------------PageRank in processing-----------------------------------')
    start_time = time.time()
    PR = nx.pagerank(G,alpha=0.8)
    end_time = time.time()
    print(f'pagerank running time is {end_time-start_time}')
    print('--------------------PageRank processing finished-----------------------------------')
    return PR
#####################################################################################

class MyThread(threading.Thread):

    def __init__(self,func,args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.results = self.func(self.args)

    def get_results(self):
        return self.results

def thread_topological_index(G):
    funcs = [tc_func,anc_func,cc_func,bec_func,dc_func,pr_func]
    threads = []
    for func in funcs:
        threads.append(MyThread(func,G))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    tc = threads[0].get_results()
    anc = threads[1].get_results()
    cc = threads[2].get_results()
    bec = threads[3].get_results()
    dc = threads[4].get_results() #测试一下degree Centrality
    pr = threads[5].get_results() #测试一下pageRank
    
    return tc,anc,cc,bec,dc,pr



if __name__ == '__main__':
    # 测试用
    test_G = nx.Graph()
    test_G.add_edges_from([(1,3,{'weight':1}),(2,3,{'weight':1}),(2,4,{'weight':2}),
                               (1,4,{'weight':2}),(3,5,{'weight':1}),(2,5,{'weight':2}),
                               (3,4,{'weight':0}),(3,6,{'weight':1}),(1,6,{'weight':2})])
    # 测试用
    tc,anc,cc,bec,dc,pr = thread_topological_index(test_G)
    print(tc)
    print(anc)
    print(cc)
    print(bec)


    # 测试用
    # tc,anc,cc,bec = topological_index(test_G)
    # print(tc)
    # print(anc)
    # print(cc)
    # print(bec)
    

