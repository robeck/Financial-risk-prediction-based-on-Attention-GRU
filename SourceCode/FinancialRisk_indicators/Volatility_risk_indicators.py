''':arg
@Author： haozhi chen
@Dates： 2023-03
@Target : 汇总Beta，volatility，turnover

'''
import pandas as pd
import numpy as np

def merge_beta_vol_turnover_process():
    # 初始化参数
    monthvolbetadata = pd.DataFrame()

    # 初始化读取相应的文件
    dfvol = pd.read_csv('~/financial_spillover_network/Data/RiskData/riskparms1.csv')
    dfturn = pd.read_csv('~/financial_spillover_network/Data/RiskData/LIQ_TOVER_M.csv')

    print(dfvol) # volatility 和 beta 数据
    "数据重构，必要股票list提取"
    dfvol['monthdate'] = [date[0:7] for date in dfvol.Trddt.tolist()]
    stocklist = list(set(dfvol.Stkcd.tolist())) # 不重复的list

    "vol beta的数据月均值计算"
    for stock in stocklist:
        stock_data = dfvol[dfvol['Stkcd']==stock].drop(['Stkcd','Trddt','Cor'],axis=1)
        stock_data_group = stock_data.groupby('monthdate').mean() # 按照monthdate分组，并且求每组内，每列的均值
        # 绘制展示（可以不用）
        # stock_data_group.plot()
        # plt.title(f'the risk indicators of {stock} ')
        # plt.show()
        stock_data_group['Stkcd'] = stock
        monthvolbetadata = pd.concat([monthvolbetadata,stock_data_group],axis=0)

    monthvolbetadata = monthvolbetadata.reset_index()
    monthlydatavol = monthvolbetadata[['monthdate','Volatility','Stkcd']] # covar的数据
    monthlydatadbeta = monthvolbetadata[['monthdate','Beta','Stkcd']] # dcovar的数据
    "分别输出"
    monthlydatavol.to_csv('~/financial_spillover_network/ResData/Risk/VolMonthRes.csv',index=False)
    monthlydatadbeta.to_csv('~/financial_spillover_network/ResData/Risk/BetaMonthRes.csv',index=False)
    print(monthvolbetadata)
    print(monthlydatavol)
    print(monthlydatadbeta)

    "turnover数据的处理"
    monthturndata = dfturn.rename(columns = {'Trdmnt':'monthdate','ToverOsM':'Turnover'})
    monthturndata.to_csv('~/financial_spillover_network/ResData/Risk/TurnMonthRes.csv',index=False)
    print(monthturndata)

    return None



if __name__ == '__main__':
    merge_beta_vol_turnover_process()