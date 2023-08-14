''':arg


Targte: 这里进行BDS和KS检验

'''
from statsmodels.tsa.stattools import bds
import scipy.stats as stats
import pandas as pd
import numpy as np

"检验金融指标和网络拓扑指标分布关系"
def ks_test():
    risknamelist = ['CoVar', 'dCoVar', 'MES', 'Beta', 'Vol', 'Turn', 'Cor', 'Illiq']
    riskindlist = ['CoVaR', 'dCoVaR', 'MES', 'Beta', 'Volatility', 'Turnover', 'Cor', 'Illiq']
    for i,riskname in enumerate(risknamelist):
        "参数设置"
        tlist,plist = [],[]
        "数据读取，必要参数提取"
        df = pd.read_csv(f'~/financial_spillover_network/ResData/Risk/{riskname}Month_mergetopo_Res.csv')
        print(df.head())
        col = riskindlist[i]
        stocklist = list(set(df.Stkcd.tolist()))  # 股票序列

        topos = ['total_connectedness', 'connectivity', 'closeness_centrality', 'betweenness_centrality',
                 'degree_centrality', 'pagerank']
        for top in topos:
            dfstock = df[[top,col,'Stkcd','monthdate']]
            manwhittlist, manwhiplist = [], []
            ksttlist, ksplist = [], []
            peartlist, pearplist = [],[]
            for stock in stocklist:
                stock_data = dfstock[dfstock.Stkcd == stock].set_index('monthdate')  # 日期边索引
                stock_data.drop(['Stkcd'], axis=1, inplace=True)  # 提出不要的stkcd号
                x = stock_data[col].tolist()
                y = stock_data[top].tolist()
                #  Mann-Whitney检验
                res1 = stats.mannwhitneyu(x, y)
                manwhittlist.append(res1[0])
                manwhiplist.append(res1[1])
                # K-S检验,检验两总体分布是否相同，在信用评级模型中可以验证模型对违约对象的区分能力
                res2 = stats.ks_2samp(x, y)
                ksttlist.append(res2[0])
                ksplist.append(res2[1])
                # 线性相关性检验
                res3 = stats.pearsonr(x,y)
                peartlist.append(res3[0])
                pearplist.append(res3[1])

            # print(f'风险指标{col}和拓扑指标{top}的Mann-Whitney检验 test statistics : {np.mean(manwhittlist)}')
            # print(f'风险指标{col}和拓扑指标{top}的Mann-Whitney检验 p value : {np.mean(manwhiplist)}')
            # print(f'风险指标{col}和拓扑指标{top}的K-S检验 test statistics : {np.mean(ksttlist)}')
            # print(f'风险指标{col}和拓扑指标{top}的K-S检验 p value : {np.mean(ksplist)}')
            print(f'风险指标{col}和拓扑指标{top}的相关性检验 test statistics : {np.mean(peartlist)}')
            print(f'风险指标{col}和拓扑指标{top}的相关性检验 p value : {np.mean(pearplist)}')

    return None


"检验金融指标是否是非线性趋势"
''':arg
1)循环每个金融指标
2）循环每一个公司
3）进行整体的均值计算，得到每个金融指标的t，p值
'''
def bds_test():
    risknamelist = ['CoVar', 'dCoVar', 'MES', 'Beta', 'Vol', 'Turn', 'Cor', 'Illiq']
    riskindlist = ['CoVaR', 'dCoVaR', 'MES', 'Beta', 'Volatility', 'Turnover', 'Cor', 'Illiq']
    for i,riskname in enumerate(risknamelist):
        "参数设置"
        tlist,plist = [],[]
        "数据读取，必要参数提取"
        df = pd.read_csv(f'~/financial_spillover_network/ResData/Risk/{riskname}MonthRes.csv')
        col = riskindlist[i]
        stocklist = list(set(df.Stkcd.tolist()))  # 股票序列

        for stock in stocklist:
            stock_data = df[df.Stkcd == stock].set_index('monthdate')  # 日期边索引
            stock_data.drop(['Stkcd'], axis=1, inplace=True)  # 提出不要的stkcd号
            x = stock_data[col].tolist()
            res = bds(x, max_dim=2, epsilon=None, distance=1.5)  # 输出包括了 bds_stat,pvalue
            tlist.append(res[0])
            plist.append(res[1])
        print(f'风险指标{col}的BDS检验 test statistics : {np.mean(tlist)}')
        print(f'风险指标{col}的BDS检验 p value : {np.mean(plist)}')

if __name__ == '__main__':
    ks_test()
    bds_test()