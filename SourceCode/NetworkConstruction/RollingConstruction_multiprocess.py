'''
@Author: haozhi chen
@Date: 2022-09
@Target：对需要进行dcc运算的数据进行初步处理，并且计算其结果

!!注意：我们又不去做预测，我们只需要每年构建一个网络不就行了？
1）我们是每年构建网络！
2）滚动进行预测（这里不需要考虑）

想法：2023-02 不予实施。存在网络构建复杂，ST样本难以复用的问题
有一个新的网络构建思路：使用预测期的股票标的，生成T-3期的网络。间隔一年逐步循环。
因此，我们需要重新编写一个新的网络生成逻辑，并且存储到新的txt文件中
'''
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import time
from tqdm import tqdm
from multiprocessing import Process
# from SourceCode.NetworkConstruction import Network,RelationConstruction
from SourceCode.NetworkConstruction.LayerDataProcess import read_data, data_preprocess
from SourceCode.NetworkConstruction.NetworkConstruct import layer_construct_

''':这里是Network构建的主函数
我们撰写不同class的作用各不相同
1）这里是为了存储生成的网络
2）——networkconstruction部分则是 不考虑滚动的，单期的网络
3）——relationconstruction部分则是，根据计算的dcc统计出网络节点间的relation系数

PS：
1) 我们的撰写结构应该是非常明确的！
2）我们在多进程结构下，不需要循环时间！！ 这点非常重要
'''
def main(filename,window_dates):
    filelists = ['~/financial_spillover_network/Data/TradingData/TargetdayTrading.csv',  # 收益率数据
                 '~/financial_spillover_network/Data/Research_target_adj.csv',  # 研究标的
                 ]
    datalist = read_data(filelists)

    networklist = []
    stocklist_period, data_period = data_preprocess(datalist, window_dates)
    "检查一下数据处理的输出"
    # print(data_period)
    # print(len(stocklist_period))

    network = layer_construct_(data_period, stocklist_period, window_dates)
    networklist.append(network) # 这里必须用list来存储，因为只有list才能被pickle模块存储到txt文件中

    print(f'测试网络节点：{network.getNodes()}')

    # 将graph全部对象的list全部存入txt文件中
    # 使用pickle dump和load函数，实际上是对json结构数据的一种扩展
    # 值得注意的是。pickle必须使用绝对路径，而不能是相对路径
    f = open(filename, 'wb')
    pickle.dump(networklist, f, 0)  # 因为只有一个网络，那么这个网路数据输出到txt文件中

    return None


''':这里的multiprocess进行真实部署
1)配置参数
2）进行调度
'''
def run_process(arglist):
    for i in range(len(arglist)):
        args = arglist[i]
        process = Process(target=main, args=args)
        process.start()
        time.sleep(10)
        print('processes are working')

"多进程调度"
''': 这里存在一些思考
1）我们需要一定时间的数据才能计算 dcc-garch 相关性系数
2）1-3 月数据 计算出3月的网络，2-4月数据 计算出 5月的网络。。。一次类推
'''
def invoke_process():
    "滑动窗口制作时间间隔的list"
    def sliding_window(seq, window_size):
        for i in range(len(seq) - window_size + interval):
            yield seq[i:i + window_size]
    "时间范围设置"
    years = ['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    window_size = 3
    interval = 1
    "1.多进程参数配置"
    yearmonthlist = [int(year+month) for year in years for month in months] # 列表推导式
    # yearmonthlist = yearmonthlist[2:] # 从2013-03到2022-12
    savedfilelist = []
    savedwindowlist = []
    "滑动窗口进行输出"
    for window in sliding_window(yearmonthlist,3):
        print(window)
        filename = f'/home/haozhic/financial_spillover_network/ResData/Networks/network_{window[0]}_{window[2]}.txt'
        savedfilelist.append(filename)
        savedwindowlist.append(window)

    arglist = list(zip(savedfilelist, savedwindowlist))
    "2.调配多进程"
    run_process(arglist)

    return None

if __name__ == '__main__':
    # tags = 'new' # 新数据选择标准下的实验，因此要是用新tags。不需要的话可以删除还原
    invoke_process()
