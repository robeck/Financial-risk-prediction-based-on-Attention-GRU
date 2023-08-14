'''
@Author: haozhi chen
@Date: 2022-09
@Target: 对节点之间的关系（dcc-garch计算的值）进行处理


'''

import pandas as pd
import numpy
import time
import logging
from SourceCode.DCC_GARCH.dcc_garch import dcc_relation  # 计算dcc的模型




''':使用dcc计算没一对数据集之间的系数
1）处理匹配两组数据，保证一致性
2）计算两组数据的dcc值
'''
def relation_extract(data,stocki,stockj,logger):
    # 初始设定一些常用参数
    data_i = data[data['Stkcd']==stocki]
    data_j = data[data['Stkcd']==stockj]
    data_i = data_i[['Date','Return']].rename(columns={'Return':'Returni'}).set_index('Date')
    data_j = data_j[['Date','Return']].rename(columns={'Return':'Returnj'}).set_index('Date')
    "1. merge一下，将时间匹配"
    # df_merge = pd.merge(data_i,data_j,left_on='Date',right_on='Date',how='inner')
    "merge outer的方案"
    df_merge = pd.merge(data_i, data_j, left_on='Date', right_on='Date', how='outer')
    df_merge.fillna(df_merge.mean(),inplace=True)

    "2. 在拆分，这样时间上就是一致的数据"
    df_returni = df_merge['Returni']
    df_returnj = df_merge['Returnj']


    "3. 计算dcc的值"
    try:
        r = dcc_relation(df_returni,df_returnj)
        print("The spillover information are  estimated")
        logger.info("The spillover information are  estimated")
    except (ValueError,numpy.linalg.LinAlgError):
        print("Face the singular matrix")
        logger.warning("Face the singular matrix")
        r = 0

    return r


''':这里汇总全部的溢出数据（dcc relation）
1）循环每个公司，构建公司对
2）每组数据，调用relation_extraction（）
3）汇总数据
'''
def layer_relation(data,stocklist,logger):
    # 初始设定一些参数
    nx_relationship, nx_relationship_weight = [],[]
    LengthStocklist = len(stocklist)


    for i in range(LengthStocklist): # 第一只股票
        for j in range(LengthStocklist): # 再一次全部搜索，形成全连接矩阵
            stocki,stockj = stocklist[i],stocklist[j]
            # print(f'work on stock {stocki} and stock{j}')
            logger.info(f"work on stock {stocki} and stock{j}") #使用logger记录必要的内容
            if i==j:
                weightij = 0
                weightij_dict = {'weight':weightij}
            else:
                weightij = relation_extract(data,stocki,stockj,logger)
                weightij_dict = {'weight':weightij}
            nx_relationship.append((stocki,stockj)) #保存关系list
            nx_relationship_weight.append((stocki,stockj,weightij_dict))# 保存关系list和权重

    ''':注意
    1. 我们的输出中包含很多weight = 0的值，这是因为
        （1）对角线的都是0
        （2）无法计算的都为0
    '''
    logger.info("Finish") #最后的logger记录
    return nx_relationship,nx_relationship_weight


if __name__ == '__main__':
    layer_relation()