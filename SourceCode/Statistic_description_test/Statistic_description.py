''':arg


Targets: 这里进行数据的描述性统计

'''
import pandas as pd
import numpy as np

def risk_indicators_statistic():
    risknamelist = ['CoVar', 'dCoVar', 'MES', 'Beta', 'Vol', 'Turn', 'Cor', 'Illiq']
    for i,riskname in enumerate(risknamelist):
        df = pd.read_csv(f'~/financial_spillover_network/ResData/Risk/{riskname}MonthRes.csv')
        print(df.describe())

    return None



def network_indicators_statistic():
    df = pd.read_csv('~/financial_spillover_network/ResData/Indicators/Network_indicators.csv')
    netstatisdescr = df.describe()
    netstatisdescr.to_csv('~/financial_spillover_network/ResData/Indicators/Network_indicators_statistic.csv',index=True)
    print(df.describe())

    return None


if __name__ == '__main__':
    # risk_indicators_statistic()
    network_indicators_statistic()