'''
@Author: haozhi chen
@Date: 2022-09
@Target：对需要进行dcc运算的数据进行初步处理，并且计算其结果


'''

import pandas as pd
import numpy as np
import random


'''
数据的读取
'''
def read_data(filelists):
    stocktrading, stocktarget= filelists
    df_trading = pd.read_csv(stocktrading) # 交易数据
    df_target = pd.read_csv(stocktarget) # 研究标的数据
    ''' 交易数据的参数
    'Stkcd', 'Trddt','Return','Date
    '''
    datalist = [df_trading, df_target]

    return datalist

''':数据的预先处理
!! 我们直接按照要一次性完成的方案进行数据处理的撰写
'''
def data_preprocess(datas,dates):
    # 为数字变量命名
    start_date,middle_date,end_date = dates

    df_stock,df_targets = datas[0],datas[1]

    df_stock = df_stock[['Stkcd','Return','Date']] # 选取必要的数据列
    "1.处理时间维度"
    df_stock_trade_test = df_stock[(df_stock['Date']>=start_date) & (df_stock['Date']<=end_date)] # 交易数据截取时间

    "2.交易数据处理"
    # df_stock_trade_test['return'] = df_stock_trade_test.Wclsprc.pct_change() # 收益率
    # df_stock_trade_test['log_return'] = np.log(df_stock_trade_test['Wclsprc'] / df_stock_trade_test.Wclsprc.shift(1))
    final_df_stock_trade_test = df_stock_trade_test.reset_index().drop('index',axis=1)
    final_df_stock_trade_test.sort_values(['Stkcd','Date'],inplace=True)
    print(final_df_stock_trade_test)

    "3. 根据Dataprocess中，我们对研究对象的筛选按，这里只要进行数据匹配，提取研究对象的交易数据即可！"
    df_targets_list = [int(stock) for stock in df_targets.Stkcd.tolist()] # 提取其中的股票list
    df_targets_list.sort() # 排序
    rows = [row for row in final_df_stock_trade_test.index if final_df_stock_trade_test.iat[row,0] in df_targets_list] # 提取交易数据中stock在list中的
    final_df_stock = final_df_stock_trade_test.iloc[rows,:] # 对应行的数据

    "4. 克服stocklist不一致长的问题"
    final_stock_list = final_df_stock.Stkcd.tolist()
    stock_list = [stock for stock in df_targets_list if stock in final_stock_list] # 保持上市公司 和 有交易信息的公司 数量和代码一致
    "测试用"
    return stock_list,final_df_stock




if __name__ == '__main__':
    pass
    # Stock_STinfo_year_file = '~/ListedCompany_risk/Data/StockRiskWarning_processed.csv'  # 包含了ST信息的上市公司11年-22年的数据
    # Stock_daily_trade_info = '~/ListedCompany_risk/Data/StockWeekly.csv'  # 上市公司的全部交易数据（周度，还是日度，需要考虑）
    # Stock_Market_value = '~/ListedCompany_risk/Data/StockMarketValue.csv'  # 上市公司是指数据
    # filelist = [Stock_STinfo_year_file,Stock_daily_trade_info,Stock_Market_value]
    #
    # for date in range(2015,2022):
    #     data = read_data(filelist)
    #     df_total_list,final_df_stock = data_preprocess(data,date)
    #     print(df_total_list)
    #     L = list(set(final_df_stock.Stkcd.tolist()))
    #     L.sort()
    #     print(L)
    #
    #     print(len(df_total_list))
    #     print(len(L))


