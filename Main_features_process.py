''':arg



'''
import pandas as pd
import numpy as np
from multiprocessing import Process
from SourceCode.NetworkConstruction.TopoImplement import Implementation

"拓扑数据的获取"
def topo_features_process(start_date,end_date,impletags):
    filename = f'/home/haozhic/financial_spillover_network/ResData/Networks/network_{start_date}_{end_date}.txt'
    res = Implementation.single_date_implement(filename,impletags)
    return res

"单独对网络的拓扑特征进行提取，并存储到文件中"
def single_topo_features_process(windows,impletags):
    print('---------------- The current integrated research ------------------------')
    topolres = topo_features_process(windows[0], windows[2], impletags)
    topolres.fillna(topolres.mean(),inplace=True) # 填充nan为均值
    print(topolres)
    topolres.to_csv(f'~/financial_spillover_network/ResData/Networks/topological_features_{windows[0]}_{windows[2]}.csv',index=False)

    return None

"主要特征的合并"
'''
1）拓扑数据
2）风险数据
'''
def main_feature_merge():
    print('---------------- The current integrated research ------------------------')

    "1) Network拓扑指标汇总"
    finaltopdata = pd.DataFrame()
    "滑动窗口制作时间间隔的list"
    def sliding_window(seq, window_size):
        for i in range(len(seq) - window_size + interval):
            yield seq[i:i + window_size]

    "时间范围设置"
    years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    window_size = 3
    interval = 1
    "多进程时间配置"
    yearmonthlist = [int(year + month) for year in years for month in months]  # 列表推导式
    "循环读取，汇总"
    for window in sliding_window(yearmonthlist,3):
        topdata = pd.read_csv(f'~/financial_spillover_network/ResData/Networks/topological_features_{window[0]}_{window[2]}.csv')
        # ['total_connectedness', 'connectivity','closeness_centrality', 'betweenness_centrality','degree_centrality','pagerank']
        # topdata = topdata[['closeness_centrality', 'betweenness_centrality','degree_centrality','pagerank','Stkcd']] # 只有中心性的
        topdata['monthdate'] = window[2]
        topdata['monthdate'] = [str(date)[0:4]+'-'+str(date)[4:] for date in topdata.monthdate.tolist()] # 日期从201301 变成 2013-01
        finaltopdata = pd.concat([finaltopdata,topdata],axis=0)
    finaltopdata = finaltopdata.reset_index().drop('index',axis=1) # 全部公司，全部时间
    finaltopdata.to_csv('~/financial_spillover_network/ResData/Indicators/Network_indicators.csv',index=False)
    print(finaltopdata)

    "2) 风险指标的提取"
    risknamelist = ['CoVar','dCoVar','MES','Beta','Vol','Turn','Cor','Illiq']
    riskindlist = ['CoVaR','dCoVaR','MES','Beta','Volatility','Turnover','Cor','Illiq']
    for i,riskname in enumerate(risknamelist):
        df = pd.read_csv(f'~/financial_spillover_network/ResData/Risk/{riskname}MonthRes.csv')
        riskmerge = pd.merge(df, finaltopdata, how='inner', on=['Stkcd', 'monthdate'])
        # riskmerge.fillna(covarmerge.mean()) # 填补nan
        riskmerge.dropna(inplace=True)  # 删除nan
        "这里可以调整选择的指标"
        # ['total_connectedness', 'connectivity','closeness_centrality', 'betweenness_centrality','degree_centrality','pagerank']
        riskmerge = riskmerge[['Stkcd', 'monthdate',riskindlist[i],'total_connectedness', 'connectivity','closeness_centrality', 'betweenness_centrality','degree_centrality','pagerank']]

        # 小测每一个stock的
        # stocklist = list(set(covarmerge.Stkcd.tolist()))
        # for stock in stocklist:
        #     datastock = covarmerge[covarmerge['Stkcd']==stock]
        #     print(datastock)

        print(riskmerge)
        riskmerge.to_csv(f'~/financial_spillover_network/ResData/Risk/{riskname}Month_mergetopo_Res.csv',index=False)


    return None


"多线程的启动"
''':arg
1）使用rolling-window，读取时间
2）针对每个时间进行多线程工作-提取拓扑数据
'''
def multiprocess():
    # 部分参数预设
    netimpletags = 'Cut_zero' # 缩减权重为0的边

    "滑动窗口制作时间间隔的list"
    def sliding_window(seq, window_size):
        for i in range(len(seq) - window_size + interval):
            yield seq[i:i + window_size]

    "时间范围设置"
    years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    window_size = 3
    interval = 1
    "多进程时间配置"
    yearmonthlist = [int(year + month) for year in years for month in months]  # 列表推导式

    "多进程进行，加快计算"

    #########################################################################
    "多进程单独存储和提取网络拓扑特征"
    for window in sliding_window(yearmonthlist,3):
        args = window,netimpletags, # 每一年的参数
        process = Process(target=single_topo_features_process, args=args)
        process.start()
        print(f'processes {process.name} are working')
    ##########################################################################

    return None


if __name__ == '__main__':
    # 提取网络拓扑数据
    # multiprocess()

    # 汇总网络数据，风险数据
    main_feature_merge()