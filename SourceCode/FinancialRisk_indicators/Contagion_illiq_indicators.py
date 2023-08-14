''':arg
@Author： haozhi chen
@Dates： 2023-03
@Target : 计算corr, illiq等指标

'''

import pandas as pd
import numpy as np

def merge_cor_illiq_process():
    # 初始化参数
    monthcordata = pd.DataFrame()

    # 初始化读取相应的文件
    dfcor = pd.read_csv('~/financial_spillover_network/Data/RiskData/riskparms1.csv')
    dfilliq = pd.read_csv('~/financial_spillover_network/Data/RiskData/LIQ_AMIHUD_M.csv')

    print(dfcor) # volatility 和 beta 数据
    "数据重构，必要股票list提取"
    dfcor['monthdate'] = [date[0:7] for date in dfcor.Trddt.tolist()]
    stocklist = list(set(dfcor.Stkcd.tolist())) # 不重复的list

    "cor的数据月均值计算"
    for stock in stocklist:
        stock_data = dfcor[dfcor['Stkcd']==stock].drop(['Stkcd','Trddt','Volatility','Beta'],axis=1)
        stock_data_group = stock_data.groupby('monthdate').mean() # 按照monthdate分组，并且求每组内，每列的均值
        # 绘制展示（可以不用）
        # stock_data_group.plot()
        # plt.title(f'the risk indicators of {stock} ')
        # plt.show()
        stock_data_group['Stkcd'] = stock
        monthcordata = pd.concat([monthcordata,stock_data_group],axis=0)

    monthcordata = monthcordata.reset_index()
    monthcordata.to_csv('~/financial_spillover_network/ResData/Risk/CorMonthRes.csv',index=False)
    print(monthcordata)

    "Illiq数据的处理"
    monthilliqdata = dfilliq.rename(columns = {'Trdmnt':'monthdate','ILLIQ_M':'Illiq'})
    monthilliqdata.to_csv('~/financial_spillover_network/ResData/Risk/IlliqMonthRes.csv',index=False)
    print(monthilliqdata)

    return None



if __name__ == '__main__':
    merge_cor_illiq_process()