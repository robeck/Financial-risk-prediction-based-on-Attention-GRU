'''
@Author: haozhi chen
@Date: 2022-09
@Target：实现对网络结构中拓扑数据的提取，汇总

'''

import pandas as pd
import networkx as nx
import numpy as np
import pickle
import MyError
from SourceCode.NetworkConstruction.TopoImplement import Topological_index

'''实现读取网络和提取拓扑指标
一次性全部读取，并且汇总成可以输出的文件
1）读取网络
2）提取指标
'''
def comprehensive_impelmentation(filenames,impletag):
    '''
    :param filenames:所有网络的文件名，这里不需要指明数量。我们用list存储全部的即可
    :return:
    '''
    "初始一些存储变量"
    networkslist = [] # 存储所有时期的网络！
    datelists = [] # 存储每一个网络下的日期
    individual_topologicalindex = pd.DataFrame()

    for filename in filenames:
        try:
            f = open(filename,'rb')
            networks = pickle.load(f)[0] # network都是读取的networks的[0]位置的数据，这里提取第一个元素即可
            date = networks.getDate()
            "数据存储:网络，日期！"
            networkslist.append(networks)
            datelists.append(date)
        except:
            raise MyError.Myexception('网络txt文件读取问题')

    "1. 我们开始计算+处理，统计每一年（年度网路）的拓扑数据"
    for i,network in enumerate(networkslist): # 逐个处理网络
        "初始一些参数"
        temp_topologicalindex = pd.DataFrame() # 一个临时存储每一年的拓扑数据
        date = datelists[i] # 日期
        print(f'我们正在对 {date} 的网络数据进行处理')
        
        "1.1. 网络修剪（删减）"
        if impletag == 'Normal':
            networkG = network.get_network() # 初始化网络，不经过任何处理的
        elif impletag == 'Cut_zero':
            networkG = network.cut_zeroweight_edges().getNetwork()
        elif impletag == 'PMFG':
            networkG = network.cut_zeroweight_edges().compute_PMFG().getNetwork()
        elif impletag == 'MST':
            networkG = network.cut_zeroweight_edges().compute_MST().getNetwork()
        else:
            raise MyError.Myexception('不存在的网络删减算法，检查 impletag 输入')

        "1.2 每年网络中每个节点（公司）的拓扑指标统计"
        TC, ANC, CC, BEC = Topological_index.topological_index(networkG)
        parameters = ['total_connectedness','connectivity','closeness_centrality','betweenness_centrality']
        lengthtc,lengthanc,lengthcc,lengthbec = len(TC),len(ANC),len(CC),len(BEC) # 统计每一个输出拓扑数据的长度，这个长度表示其中的节点数
        "字典化数据"
        if lengthtc==lengthanc==lengthcc==lengthbec: # 长度一致一次性处理
            Totaltopindex = {'Stkcd':[int(x) for x in list(TC.keys())],
                             parameters[0]:list(TC.values()),
                             parameters[1]:list(ANC.values()),
                             parameters[2]:list(CC.values()),
                             parameters[3]:list(BEC.values())}
            temp_topologicalindex = pd.DataFrame(Totaltopindex)
        else: # 不一致的情况要进行单独生成dataframe并汇总
            dftc = pd.DataFrame({'Stkcd':[int(x) for x in list(TC.keys())],parameters[0]:list(TC.values())})
            dfanc = pd.DataFrame({'Stkcd':[int(x) for x in list(ANC.keys())],parameters[1]:list(ANC.values())}) 
            dfcc = pd.DataFrame({'Stkcd':[int(x) for x in list(CC.keys())],parameters[2]:list(CC.values())})
            dfbec = pd.DataFrame({'Stkcd':[int(x) for x in list(BEC.kets())],parameters[3]:list(BEC.values())})
            temp_topologicalindex = pd.merge(dftc,pd.merge(dfanc,pd.merge(dfcc,dfbec,how='inner',on='Stkcd'),
                                                            how='inner',on='Stkcd'),
                                              how='inner',on='Stkcd')
        individual_topologicalindex = pd.concat([individual_topologicalindex.copy(),temp_topologicalindex],axis=0)

    "计算出来的拓扑数据进行存储即可，也可以通过其他函数调用这里的输出，都是可以的！"
    # individual_topologicalindex.to_csv('~/ListedCompany_risk/Data/Outputdata/individual_topological_index.csv')
    return individual_topologicalindex


'''通过日期，时间数据调用
单个时间的数据读取，符合我们 main_function 的逻辑
'''
def single_date_implement(filename,impletag):
    "初始一些存储变量"
    individual_topologicalindex = pd.DataFrame()

    try:
        f = open(filename, 'rb')
        network = pickle.load(f)[0]  # network都是读取的networks的[0]位置的数据，这里提取第一个元素即可
        date = network.getDate()
        "数据存储:网络，日期！"
    except:
        raise MyError.Myexception('网络txt文件读取问题')

    "1. 我们开始计算+处理，统计每一年（年度网路）的拓扑数据"
    "初始一些参数"
    temp_topologicalindex = pd.DataFrame()  # 一个临时存储每一年的拓扑数据
    print(f'我们正在对 {date} 的网络数据进行处理')

    "1.1. 网络修剪（删减）"
    if impletag == 'Normal':
        networkG = network.get_network()  # 初始化网络，不经过任何处理的
    elif impletag == 'Cut_zero':
        networkG = network.cut_zeroweight_edges().getNetwork()
    elif impletag == 'PMFG':
        networkG = network.cut_zeroweight_edges().compute_PMFG().getNetwork()
    elif impletag == 'MST':
        networkG = network.cut_zeroweight_edges().compute_MST().getNetwork()
    else:
        raise MyError.Myexception('不存在的网络删减算法，检查 impletag 输入')

    "1.2 每年网络中每个节点（公司）的拓扑指标统计"
    # TC, ANC, CC, BEC = Topological_index.topological_index(networkG) # 单线程运算每一个指标
    TC, ANC, CC, BEC, DC, PR = Topological_index.thread_topological_index(networkG) # 使用多线程方法计算每一个指标
    print('Topological indicators processings are finished !')
    "参数提前给名字"
    parameters = ['total_connectedness', 'connectivity', 'closeness_centrality', 'betweenness_centrality',
                  'degree_centrality','pagerank']
    lengthtc, lengthanc, lengthcc, lengthbec,lenthdc,lengthpr = len(TC), len(ANC), len(CC), len(BEC), len(DC),len(PR)  # 统计每一个输出拓扑数据的长度，这个长度表示其中的节点数
    "字典化数据"
    if lengthtc == lengthanc == lengthcc == lengthbec == lenthdc == lengthpr:  # 长度一致一次性处理
        print('The same length of topological indicators processing')
        Totaltopindex = {'Stkcd': [int(x) for x in list(TC.keys())],
                         parameters[0]: list(TC.values()),
                         parameters[1]: list(ANC.values()),
                         parameters[2]: list(CC.values()),
                         parameters[3]: list(BEC.values()),
                         parameters[4]:list(DC.values()),
                         parameters[5]:list(PR.values())}
        temp_topologicalindex = pd.DataFrame(Totaltopindex)
    else:  # 不一致的情况要进行单独生成dataframe并汇总
        print('The different length of topological indicators processing')
        dftc = pd.DataFrame({'Stkcd': [int(x) for x in list(TC.keys())], parameters[0]: list(TC.values())})
        dfanc = pd.DataFrame({'Stkcd': [int(x) for x in list(ANC.keys())], parameters[1]: list(ANC.values())})
        dfcc = pd.DataFrame({'Stkcd': [int(x) for x in list(CC.keys())], parameters[2]: list(CC.values())})
        dfbec = pd.DataFrame({'Stkcd': [int(x) for x in list(BEC.kets())], parameters[3]: list(BEC.values())})
        dfdc = pd.DataFrame({'Stkcd': [int(x) for x in list(DC.kets())], parameters[3]: list(DC.values())})
        dfpr = pd.DataFrame({'Stkcd': [int(x) for x in list(PR.kets())], parameters[3]: list(PR.values())})
        temp_topologicalindex = pd.merge(dftc, pd.merge(dfanc, pd.merge(dfcc, pd.merge(dfbec,pd.merge(dfdc,dfpr,how='inner',on='Stkcd'),
                                                                                       how='inner',on='Stkcd'),
                                                                        how='inner', on='Stkcd'),
                                                        how='inner', on='Stkcd'),
                                         how='inner', on='Stkcd')
    individual_topologicalindex_df = pd.concat([individual_topologicalindex.copy(), temp_topologicalindex], axis=0)

    return individual_topologicalindex_df


if __name__ == '__main__':
    # 测试！
    filenames = []
    impletag = 'Cut_zero'
    ranges = np.arange(2011,2022)

    "全部日期数据的测试"
    # for year in ranges:
    #     filename = f'/home/haozhic2/ListedCompany_risk/Data/Networkdata/network_{year}.txt'
    #     filenames.append(filename)
    # comprehensive_impelmentation(filenames,impletag)

    "某一特定日期数据的测试"
    filename = '/home/haozhic2/ListedCompany_risk/Data/Networkdata/network_2015.txt'
    df = single_date_implement(filename,impletag)
    print(df)