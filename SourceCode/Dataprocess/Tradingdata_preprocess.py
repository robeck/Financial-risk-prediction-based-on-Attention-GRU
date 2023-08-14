'''
@Author: haozhi chen
@Date : 2023-03
@Target : concat trading data

'''

import pandas as pd
import numpy as np

"合并交易数据：研究标的公司，2013-2022十，日度数据"
def concat():
    # 初始化数据位置
    # file1 = '~/financial_spillover_network/Data/TradingData/TRD_Dalyr.csv' # 交易数据1 #没有市值的数据
    # file2 = '~/financial_spillover_network/Data/TradingData/TRD_Dalyr1.csv' # 交易数据2 #没有市值的数据
    file1 = '~/financial_spillover_network/Data/TradingData/TRD_Dalyr2013_2018.csv' # 交易数据1
    file2 = '~/financial_spillover_network/Data/TradingData/TRD_Dalyr2018_2022.csv' # 交易数据2
    "读取数据"
    df1,df2 = pd.read_csv(file1),pd.read_csv(file2)
    "数据合并，修缮"
    df = pd.concat([df1,df2],axis=0) # 合并
    df = df.sort_values(by=['Trddt','Stkcd']) # 按时间排序
    df = df.reset_index().drop(columns=['index']) # 重置索引
    # df = df.rename(columns = {'Dretnd':'Return'}) # 日度收益重命名
    df = df.rename(columns={'Dretnd': 'Return','Dsmvtll':'Market_value'})  # 日度收益,总市值重命名

    "如果市值数据存在，对数化"
    df['Size'] = [np.log(marketcap) for marketcap in df.Market_value.tolist()] # 对数化市值
    df.drop(columns=['Market_value'],inplace=True) # 删除普通市值

    "增加一个只用年月的数据"
    MonthDates = [int(dates[0:4]+dates[5:7]) for dates in df.Trddt.tolist()] # 这里的日期产生的数据从 2013-01 变成了 201301
    df['Date'] = MonthDates # df增加一列

    "增加一个测试部分"
    file3 = '~/financial_spillover_network/Data/Research_target.csv'
    df3 = pd.read_csv(file3)
    stocks = [int(stock[0:6]) for stock in df3.Stkcd.tolist()]
    print(stocks)
    stocks_1 = [stock for stock in list(set(df.Stkcd.tolist()))]
    print(stocks_1)
    test_stock = [stock for stock in stocks if stock in stocks_1] # stock 在stocks中 也在 stocks_1中
    print(test_stock)
    print(f'stocks : {len(stocks)}, stocks_1 : {len(stocks_1)},stocks_2 : {len(test_stock)}')
    if len(stocks)==len(stocks_1)==len(test_stock):
        print('测试正确，上市公司数据齐全')
    else:
        print('数据不匹配，需要检查')

    print(df) #展示一下
    df.to_csv('~/financial_spillover_network/Data/TradingData/TargetdayTrading.csv',index=False) # 输出存储

    dfdate = df[(df['Date']>=201301) & (df['Date']<=201303)] # 测试而已
    print(dfdate)

    return df

"对研究标的数据进行重构"
def reasarch_target_reconstruct():
    df = pd.read_csv('~/financial_spillover_network/Data/Research_target.csv')
    df['Stkcd'] = [int(stock[0:6]) for stock in df.Stkcd.tolist()] # stock 转换成int
    df.to_csv('~/financial_spillover_network/Data/Research_target_adj.csv',index=False) # 输出

    return df


if __name__ == '__main__':
    concat() # 纵向合并数据
    # reasarch_target_reconstruct()