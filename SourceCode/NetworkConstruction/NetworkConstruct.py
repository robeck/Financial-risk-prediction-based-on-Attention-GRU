'''
@Author: haozhi chen
@Date: 2022-09
@Target: 实现网络构建部分


'''
import pandas as pd
import numpy as np
import networkx as nx
import MyError
import os
import logging #自撰写log
from SourceCode.NetworkConstruction.RelationConstruct import layer_relation
from SourceCode.NetworkConstruction import Network


'''
单结构下网络的生成
输入：
    1）数据集
    2）时间
    3）股票list（nodes）
输出：
    1）自撰写的Network实例
'''
def layer_construct_(data,stocklist,window_dates):
    # 配置logger
    if os.path.exists(rf'/home/haozhic/financial_spillover_network/Data/Networkdata/log_{window_dates[2]}.txt'):
        os.remove(rf'/home/haozhic/financial_spillover_network/Data/Networkdata/log_{window_dates[2]}.txt')
    else:
        pass
    "logger配置"
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(f'/home/haozhic/financial_spillover_network/Data/Networkdata/log_{window_dates[2]}.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"We are working on the data in {window_dates[2]}")  # logger 记录
    relationlist, relationlist_weight = layer_relation(data,stocklist,logger)

    "1. 生成网络"
    Gen_network = nx.Graph()
    Gen_network.add_edges_from(relationlist_weight)

    "2. 将网络部署成为class数据，便于存储，读取使用"
    nodes = stocklist
    date = window_dates[2] # date,rolling窗口的最后日期
    network = Network.networkinstance(nodes,relationlist,relationlist_weight,date,Gen_network)

    "*检查输入的nodes是否和真实部署的nodes一致，len=0表示一致"
    g = network.getNetwork()
    mergenodes = [node for node in network.getNodes() if node in g.nodes] # 我们输入的nodes 和 进行网络构建出来的nodes 交集
    if (len(mergenodes)==len(network.getNodes())) and (len(mergenodes)==len(g.nodes)): # 检查这两个nodes序列形成的交集是否和元数据保持一致，从而确保结果的正确性
        pass
    else:
        logger.warning(f"face the non match nodes on {date}")
        raise MyError.Myexception('node匹配不一致！') #唤起自己撰写的错误

    ''':return
    Network: Network的实例！而非graph！！这点你太重要了
    '''
    return network



if __name__ == '__main__':
    layer_construct_()