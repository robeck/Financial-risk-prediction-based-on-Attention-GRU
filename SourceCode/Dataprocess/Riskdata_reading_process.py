''':
@Author ： haozhi chen
@Date : 2023-03-16
@Target : 对上市公司的风险数据进行读取，处理，并进行相应的合并

'''
import pandas as pd
import numpy as np

"读取原始数据，进行整理"
def data_merge():
    # 初始化参数
    range0 = 5
    range1 = 2

    "1. 个股beta，波动，corr指标 （日）"
    temp = pd.DataFrame()
    for i in range(0,range0):
        filename = f'~/financial_spillover_network/Data/RiskData/STK_MKT_STKBTAL{i}.csv'
        df = pd.read_csv(filename)
        temp = pd.concat([temp,df],axis=0)
    parms1 = temp.reset_index().drop(columns=['index']).rename(columns={'Symbol':'Stkcd','Beta2':'Beta','Cor2':'Cor','TradingDate':'Trddt'}) # 个股的beta，volatility，correlation（和市场收益之间）
    print(parms1)
    parms1.to_csv('~/financial_spillover_network/Data/RiskData/riskparms1.csv',index=False)

    "2. 个股PB 用于计算BM（日）"
    temp2 = pd.DataFrame()
    for i in range(0,range0):
        filename = f'~/financial_spillover_network/Data/RiskData/STK_MKT_DALYR{i}.csv'
        df = pd.read_csv(filename)
        temp2 = pd.concat([temp2,df],axis=0)
    parms2 = temp2.reset_index().drop(columns=['index','ShortName']).rename(columns={'Symbol':'Stkcd','TradingDate':'Trddt'})
    parms2['BM'] = parms2.apply(lambda x: (1 / (x.PB)), axis=1)  # 计算BM
    print(parms2)
    parms2.to_csv('~/financial_spillover_network/Data/RiskData/riskparms2.csv', index=False)

    "3. 个股换手率turnover（日）"
    temp3 = pd.DataFrame()
    for i in range(0, range1):
        filename = f'~/financial_spillover_network/Data/RiskData/LIQ_TOVER_D{i}.csv'
        df = pd.read_csv(filename)
        temp3 = pd.concat([temp3, df], axis=0)
    parms3 = temp3.reset_index().drop(columns=['index']).rename(columns={'Symbol':'Stkcd','ToverOs':'Turnover'})
    print(parms3)
    parms3.to_csv('~/financial_spillover_network/Data/RiskData/riskparms3.csv', index=False)
    
    "4. 个股流动性illiquid（日）"
    temp4 = pd.DataFrame()
    for i in range(0, range1):
        filename = f'~/financial_spillover_network/Data/RiskData/LIQ_AMIHUD_D{i}.csv'
        df = pd.read_csv(filename)
        temp4 = pd.concat([temp4, df], axis=0)
    parms4 = temp4.reset_index().drop(columns=['index']).rename(columns={'Symbol': 'Stkcd'})
    print(parms4)
    parms4.to_csv('~/financial_spillover_network/Data/RiskData/riskparms4.csv', index=False)

    "5. 市场以实现波动率 marketvol"
    parms5 = pd.read_csv('~/financial_spillover_network/Data/RiskData/HF_IndexRealized.csv')
    parms5 = parms5[parms5['Indcd']==1].drop(columns=['Indcd']).rename(columns={'RV':'Marketvol'})
    print(parms5)
    parms5.to_csv('~/financial_spillover_network/Data/RiskData/riskparms5.csv', index=False)

    "6. 期限利差termspread"
    df = pd.read_csv('~/financial_spillover_network/Data/RiskData/BND_TreasYield.csv')
    df3,df10 = df[df['Yeartomatu']==0.25].reset_index().drop(['index','Yeartomatu'],axis=1),df[df['Yeartomatu']==10].reset_index().drop(['index','Yeartomatu'],axis=1)
    merge = pd.merge(df3,df10,how='inner',on='Trddt')
    merge['Termspread'] = merge['Yield_y']-merge['Yield_x'] # 期限利差10年-3月
    parms6 = merge.drop(columns=['Cvtype_x','Cvtype_y','Yield_x','Yield_y'])
    print(parms6)
    parms6.to_csv('~/financial_spillover_network/Data/RiskData/riskparms6.csv', index=False)
    
    "7. 市场收益率marketreturn"
    df = pd.read_csv('~/financial_spillover_network/Data/RiskData/TRD_Cndalym.csv')
    parms7 = df[df['Markettype']==5].reset_index().drop(columns=['index','Markettype']).rename(columns={'Cdretmdos':'Marketreturn'})  # 沪深市场收益率
    print(parms7)
    parms7.to_csv('~/financial_spillover_network/Data/RiskData/riskparms7.csv', index=False)

    return None


"1. Covar,dcovar,mes计算数据的整理"
def covar_mes_data_process():
    # 文件位置
    "个股的"
    filestocktrading = '~/financial_spillover_network/Data/TradingData/TargetdayTrading.csv' # 个股交易数据，Size
    filestockvol = '~/financial_spillover_network/Data/RiskData/riskparms1.csv' # 波动率
    filestockBM = '~/financial_spillover_network/Data/RiskData/riskparms2.csv' # BM
    filestockIlliq = '~/financial_spillover_network/Data/RiskData/riskparms4.csv' # Illiq
    "市场的"
    filestockTermspread = '~/financial_spillover_network/Data/RiskData/riskparms6.csv' # Termspread
    filestockmarketvol = '~/financial_spillover_network/Data/RiskData/riskparms5.csv' # 市场波动
    filestockmarketreturn = '~/financial_spillover_network/Data/RiskData/riskparms7.csv' # 市场收益率
    "读取文件"
    df1,df2,df3,df4,df5,df6,df7 = pd.read_csv(filestocktrading),pd.read_csv(filestockvol),pd.read_csv(filestockBM),\
                                  pd.read_csv(filestockIlliq),pd.read_csv(filestockTermspread),pd.read_csv(filestockmarketvol),\
                                  pd.read_csv(filestockmarketreturn)
    df1 = df1.drop(columns=['Date'])
    df2 = df2.drop(columns=['Beta','Cor'])
    df3 = df3.drop(columns=['PB'])
    "合并文件1（个股日度的）"
    merge1 = pd.merge(df1,pd.merge(df2,pd.merge(df3,df4,how='outer',on=['Stkcd','Trddt']),
                                   how='outer',on=['Stkcd','Trddt']),
                      how='outer',on=['Stkcd','Trddt'])
    # merge = merge1.fillna(merge1.mean())
    merge1 = merge1.dropna()
    print(merge1)
    "合并文件2（市场日度的）"
    merge2 = pd.merge(merge1,pd.merge(df5,pd.merge(df6,df7,how='inner',on='Trddt'),
                                      how='inner',on='Trddt'),
                      how='inner',on='Trddt')
    print(merge2)
    # print(merge2.columns) # ['Stkcd', 'Trddt', 'Return', 'Size', 'Volatility', 'BM', 'ILLIQ','Termspread', 'Marketvol', 'Marketreturn']

    "滞后一阶，laggedreturn"
    final_data = pd.DataFrame()
    stocklist = list(set(merge2.Stkcd.tolist()))
    for i,stock in enumerate(stocklist):
        data_stock = merge2[merge2['Stkcd']==stock]
        data_stock = data_stock.sort_values('Trddt') # 按照日期排序
        data_stock['Laggedreturn'] = data_stock.Return.shift(1) # 滞后收益率
        # 参数滞后，除了return
        data_stock.Size = data_stock.Size.shift(1)
        data_stock.Volatility = data_stock.Volatility.shift(1)
        data_stock.BM = data_stock.BM.shift(1)
        data_stock.ILLIQ = data_stock.ILLIQ.shift(1)
        data_stock.Termspread = data_stock.Termspread.shift(1)
        data_stock.Marketvol = data_stock.Marketvol.shift(1)
        data_stock.Marketreturn = data_stock.Marketreturn.shift(1)
        data_stock = data_stock.dropna()
        # 汇总
        final_data = pd.concat([final_data,data_stock],axis=0)
    
    final_data.to_csv('~/financial_spillover_network/ResData/Risk/tailrisk_proxies.csv',index=False)
    return final_data


def beta_data_process():


    return None


if __name__ == '__main__':
    # 数据读取，整理，存储
    data_merge()
    # 数据全部整合
    # covar_mes_data_process()