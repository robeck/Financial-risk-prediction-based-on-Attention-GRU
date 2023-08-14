''':arg
@Author： haozhi chen
@Dates： 2023-03
@Target : 计算CoVaR，Dcovar，MSE等指标
'''

import pandas as pd
import numpy as np
from tqdm import tqdm
from arch import arch_model
import tushare as ts

# covar的估计模型
from statsmodels.formula.api import quantreg  # 分位数回归的模型
from scipy import stats

# 引入sklearn的核密度函数
from sklearn.neighbors import KernelDensity

from matplotlib.pylab import plt # 绘图

''':arg


'''
class covar_measurement():

    def __init__(self,data,q):
        self.data = data
        self.q = q

    def CoVaR1(returns, market_returns, alpha):
        """
        Calculates the CoVaR (Conditional Value at Risk) of a portfolio given the returns of the portfolio and the market
        returns, and a specified alpha level.

        Args:
        returns (pandas.DataFrame): A pandas DataFrame containing the returns of the portfolio.
        market_returns (pandas.DataFrame): A pandas DataFrame containing the returns of the market.
        alpha (float): The alpha level at which to calculate the CoVaR.

        Returns:
        float: The CoVaR of the portfolio at the specified alpha level.
        """
        # Calculate the portfolio's VaR at the specified alpha level
        portfolio_var = returns.quantile(alpha)

        # Calculate the market's VaR at the specified alpha level
        market_var = market_returns.quantile(alpha)

        # Calculate the portfolio's expected shortfall at the specified alpha level
        portfolio_es = returns[returns <= portfolio_var].mean()

        # Calculate the market's expected shortfall at the specified alpha level
        market_es = market_returns[market_returns <= market_var].mean()

        # Calculate the CoVaR
        covar = (portfolio_es - market_es) * (portfolio_var / market_var)

        return covar

    def CoVaR2(returns, alpha, weights):
        """
        Calculate the CoVaR (Conditional Value at Risk) of a portfolio.

        Parameters
        ----------
        returns : array-like
            Returns of the portfolio.
        alpha : float
            Confidence level.
        weights : array-like
            Weights of the portfolio.

        Returns
        -------
        float
            CoVaR of the portfolio.
        """
        # Calculate the VaR (Value at Risk) of the portfolio
        VaR = stats.norm.ppf(alpha) * np.sqrt(np.dot(weights.T, np.dot(np.cov(returns), weights)))

        # Calculate the portfolio return at the VaR level
        portfolio_return_at_VaR = np.dot(weights, returns) - VaR

        # Calculate the CoVaR
        CoVaR = -np.dot(weights, np.maximum(portfolio_return_at_VaR - np.dot(weights, returns), 0)) / (1 - alpha)

        return CoVaR

    def delta_covar(returns, market_returns, alpha, window):
        """
        计算基于Adrian和Brunnermeier（2016）方法的Delta-CoVaR风险度量。

        Args:
        - returns：一个形状为(n_obs，n_assets)的二维numpy数组，包含n_assets的历史回报。
        - market_returns：一个长度为n_obs的一维numpy数组，包含市场的历史回报。
        - alpha：一个介于0和1之间的标量值，表示置信水平。
        - window：表示滚动窗口大小的整数值。

        Returns:
        - delta_covar：表示Delta-CoVaR风险度量的标量值。
        """
        n_obs, n_assets = returns.shape
        beta = np.zeros(n_assets)
        delta_covar = 0

        for i in range(window, n_obs):
            returns_window = returns[i - window:i]
            market_returns_window = market_returns[i - window:i]
            market_returns_mean = np.mean(market_returns_window)
            market_returns_std = np.std(market_returns_window)
            market_returns_zscore = (market_returns_window[-1] - market_returns_mean) / market_returns_std

            for j in range(n_assets):
                asset_returns = returns_window[:, j]
                asset_mean = np.mean(asset_returns)
                asset_std = np.std(asset_returns)
                asset_beta = np.cov(asset_returns, market_returns_window)[-1, 0] / np.var(market_returns_window)
                beta[j] = asset_beta

            asset_contributions = asset_std * beta * (market_returns_zscore - norm.ppf(alpha))

            delta_covar += np.sum(asset_contributions[asset_contributions > 0]) * np.sqrt(window)

        return delta_covar


    # (2) 动态CoVaR 和 delta CoVaR估计
    def est_delta_covar(self,system_losses_name, stock_losses_name, M, q, roll_window, method='emp'):
        '''
        :argument
            system_losses : 系统损失，其他公司损失。这里必须是Series结构，如果输入的是Dataframe，这里需要用DataFrame.system_losses
            stock_losses : 同理，这里是个体的损失
            M : 是状态变量，用来进行回归的 结构参考（“var1 + var2 + 。。）。也就是说，状态变量是string名称即可
        '''
        # 给定初始的数据集
        system_lossed = self.data[system_losses_name]
        stock_losses = self.data[stock_losses_name]

        #############################################################################
        # 分位回归模型
        # 确定分位数回归，q为分位数，systemlosses表示市场收益，roll_var表示的是滚动的个股在线价值var
        def quantile_reg(system_losses_name, roll_var_name, M, q, data):

            # 这里将系统损失（或者其他机构损失也行）和单个机构的VaR和控制变量M进行回归
            if M !=None:
                model = quantreg(f"{system_losses_name} ~ {roll_var_name} + {M}", data)
                res = model.fit(q=q)
            else:
                model = quantreg(f'{system_losses_name} ~ {roll_var_name}',data)
                res = model.fit(q=q)

            return res

        ############################################################################
        if ((method != 'emp') and (method != 'norm')):
            raise Exception('Wrong input for method')

        if (method == 'emp'):
            # rolling:
            ''':rolling
            windows:表示窗口期
            min_period:表示窗口期内最小的观测个数
            '''
            # 这一部分是为了从收益的序列中找到分位数的var，因此这里的q应该是单独的。这一步分求出来的就是分位数回归的VaR，也是单个企业的
            # 这里的工作是滚动估计，也就是得到了VaR的序列
            # 我们可以根据这个序列，对CoVaR进行进一步的估计，也就是说我们根据Adrian的公式，只需要分位估计 system_loss ~ VaR + M 即可！
            roll_var = stock_losses.rolling(window=roll_window, min_periods=int(0.8 * roll_window)).apply(lambda x: np.quantile(x, 0.05))  # 这里的q应该是var的q，而不是全部的q
            roll_median = stock_losses.rolling(window=roll_window,min_periods=int(0.8 * roll_window)).median()  # 滚动窗口期内收益的中位数
            # 为两个序列的名称重新命名
            roll_var.rename('roll_var',inplace=True)
            roll_median.rename('roll_median',inplace=True)

            self.data = pd.merge(pd.merge(self.data,roll_var,left_index=True,right_index=True,how='inner'),
                                 roll_median,left_index=True,right_index=True,how='inner')

        else:
            roll_median = stock_losses.rolling(window=roll_window, min_periods=int(0.8 * roll_window)).mean()
            std = stock_losses.rolling(window=roll_window, min_periods=int(0.8 * roll_window)).std()
            roll_var = roll_median + stats.norm.ppf(q) * std

            roll_var.rename('roll_var',inplace=True)
            roll_median.rename('roll_median',inplace=True)

            self.data = pd.merge(pd.merge(self.data,roll_var,left_index=True,right_index=True,how='inner'),
                                 roll_median,left_index=True,right_index=True,how='inner')


        # 这里两个回归的因为Y都是系统损失，所以结果都应该是CoVaR的结果
        model_q = quantile_reg(system_losses_name, 'roll_var',M, q,self.data)
        # model_median = quantile_reg(system_losses_name, 'roll_median', M, q, self.data)  # 原作者撰写
        model_median = quantile_reg(system_losses_name, 'roll_var', M, 0.5,self.data) # covar(50%)
        # import pdb;pdb.set_trace()

        covar = model_q.fittedvalues
        delta_covar = (model_q.fittedvalues - model_median.fittedvalues)
        # 为上述两个序列重新命名
        covar.rename('covar',inplace=True)
        delta_covar.rename('delta covar',inplace=True)

        return delta_covar, roll_var, covar


class mes_measurement():

    def __init__(self,data,p):
        ''':param
        data : 输入的数据（即一个固定某个公司，所有研究时期内，个体收益+市场收益数据）
        p : 分位数
        '''
        self.data = data
        self.p = p

    def set_data(self,newdata):
        self.data = newdata

    '''
    ES 的估计
        percentile: 表示的是置信水平，如99置信水平下。percentile=99 
    '''
    def ES_estimate(self):
        percentile = 1-self.p
        VaR = np.percentile(self.data,(1-percentile)*100)
        ES = self.data[self.data<=VaR].mean()  #小于VaR部分的均值
        return ES

    # 测试99%置信水平下的ES

    '''
    # 估计方法1：是要用波动率 + dcc相关系数 + 核密度函数估计的尾部期望！
    高斯核密度估计
        X ：输入的训练数据集
        kernel : 默认为噶苏思安
        bandwidth ：带宽
    '''
    def kernel_est(self,X,bandwidth, kernel='gaussian'):
        X = self.data.reshape(-1, 1)  # 输入的训练数据，转换成2d
        # X_plot = np.linspace(0, 0.1, 1000)[:, np.newaxis]  # 使用 [:,np.newaxis] 转换成 2d array
        kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)  # 高斯核密度估计
        # log_dens = kde.score_samples(X_plot)  # 返回的是点x_plot对应概率密度的log值，需要使用exp求指数还原

        # inverse_data = np.exp(log_dens)

    '''
    # 估计方法2：使用系统收益（损失）小于VaR的时期为系统风险时期，在这个时期的个体收益（损失）的均值作为MES
    '''
    def MES_est_method2(self,systemloss_name,individualloss_name,rolling_window):
        '''
            systemloss_name : 这里是系统损失的column名称
            individual_name ：这里是个体损失的column名称
        '''
        #提取对应的序列作为计算的标的
        systemloss_df = self.data[systemloss_name]
        stockloss_df = self.data[individualloss_name]

        # 循环估计这个系统收益的VaR，作为测度MES的条件期望
        systemloss_df = systemloss_df.rolling(window=rolling_window, min_periods=int(0.8 * rolling_window)).apply(lambda x: np.quantile(x, self.p))  # 这里的q应该是var的q，而不是全部的q
        systemloss_df.rename('systemvar',inplace=True)

        # merge一下原始数据和计算后VaR的数据
        data = pd.merge(self.data,systemloss_df,how='inner',left_index=True,right_index=True).drop(columns=['Stkcd']).dropna()
        data = data[data[systemloss_name]<data['systemvar']].rename(columns={individualloss_name:'MES'}).drop(columns=[systemloss_name,'systemvar'])
        # data['MES'] = -(data['MES']) # 参考文献：Yun, T., Jeong, D., & Park, S. (2019). 加负号

        return data #返回一个MES的数据Series

def CoVaR_invoke(data):
    # 真实计算部分
    '''
    1）对数据的进一步处理
    '''
    ############################################################################### 参数设定
    system_name = 'Marketreturn'
    individual_name = 'Return'
    M = 'ILLIQ + Volatility + Size + BM + Termspread + Marketvol + Laggedreturn'
    q = 0.05
    roll_window = 60
    ###############################################################################
    # 对于groupby操作来简化这里要分组计算的工作，我们还没有清楚该怎么弄
    stocks = list(set(data.Stkcd.tolist()))
    #空datafram
    Systemriskdf = pd.DataFrame()

    for i in tqdm(range(len(stocks))):
        stock = stocks[i] # stock的名称

        dfstock = data[data['Stkcd'] == stock]  #提取某只股票的数据
        model_covar = covar_measurement(dfstock,q)
        delta_CoVar,roll_VaR,CoVaR = model_covar.est_delta_covar(system_name,individual_name,M,q,roll_window)

        "重构数据的结构"
        '''
        1)选择年月日时间，还是年月的时间，取决于数据
        '''
        # datetime = [datetime[0:7] for datetime in CoVaR.index.tolist()] # 年月
        datetime = [datetime for datetime in CoVaR.index.tolist()]  # 年月日
        dcovar = [x for x in delta_CoVar.tolist()]
        var = [x for x in roll_VaR.dropna().tolist()] #var 和 CVar 用负值标识！
        covar = [x for x in CoVaR.tolist()]
        stock = [stock] * len(datetime)

        system_risk_df = pd.DataFrame({'Stkcd':stock,'dcovar': dcovar,'var':var,'covar':covar},index=datetime)

        Systemriskdf = pd.concat([Systemriskdf,system_risk_df],axis=0)
        # 迭代增加，输出一个汇总好的结果df，输出到csv文件中
    print(Systemriskdf)
    Systemriskdf.to_csv('~/financial_spillover_network/ResData/Risk/CoVarres.csv',index=True,index_label='Date') # index给与了label


    return Systemriskdf



'''
这里对上市公司的MES指标进行估计
'''
def MES_invoke(data):
    output_data = data[['Return','Marketreturn','Stkcd']]
    stocks = list(set(data.Stkcd.tolist())) #全部的公司

    # 分位数参数
    q = 0.05
    rolling_window = 60

    # 一个空datafram来存储所有的MES数据
    MESdf = pd.DataFrame()
    for i in tqdm(range(len(stocks))):
        stock = stocks[i]
        output_stock_data = output_data[output_data['Stkcd']==stock]
        model = mes_measurement(output_stock_data,q)
        MES = model.MES_est_method2('Marketreturn','Return',rolling_window) #返回数据

        "重构部分数据"
        # datetime = [x[0:7] for x in MES.index.tolist()] #只使用包含月的日期数据
        datetime = [x for x in MES.index.tolist()]  # 只使用包含年月日的日期数据

        MES['Stkcd'] = stock
        MES.index = datetime

        MESdf = pd.concat([MESdf,MES],axis=0)

    print(MESdf)
    MESdf.to_csv('~/financial_spillover_network/ResData/Risk/MESres.csv',index=True,index_label='Date')

    return MESdf

'''
数据的简单处理，输出包含必要数据的部分
    :return
    时间为index
'''
def data_process(data):
    # 数据的裁剪和处理
    splitdata = data.copy().rename(columns={'Trddt': 'Date'}).set_index('Date')  # 输出时间为index，return为序列的dataframe
    splitdata.sort_index(inplace=True)
    return splitdata

"计算的风险数据的月度话（求均值）"
def risk_monthly_process(datacovar,datames):
    # 初始化参数
    monthlydatacovar = pd.DataFrame()
    monthlydatames = pd.DataFrame()

    "数据重构，必要股票list提取"
    datacovar['monthdate'] = [date[0:7] for date in datacovar.Date.tolist()]
    datames['monthdate'] = [date[0:7] for date in datames.Date.tolist()]
    stocklist = list(set(datacovar.Stkcd.tolist())) # 不重复的list

    "covar的数据月均值计算"
    for stock in stocklist:
        stock_data = datacovar[datacovar['Stkcd']==stock].drop('Stkcd',axis=1)
        stock_data_group = stock_data.groupby('monthdate').mean() # 按照monthdate分组，并且求每组内，每列的均值
        # 绘制展示（可以不用）
        # stock_data_group.plot()
        # plt.title(f'the risk indicators of {stock} ')
        # plt.show()
        stock_data_group['Stkcd'] = stock
        monthlydatacovar = pd.concat([monthlydatacovar,stock_data_group],axis=0)

    monthlydatacovar = monthlydatacovar.reset_index().rename(columns={'dcovar':'dCoVaR','covar':'CoVaR'})
    monthlydataCoVar = monthlydatacovar[['monthdate','CoVaR','Stkcd']] # covar的数据
    monthlydatadCoVar = monthlydatacovar[['monthdate','dCoVaR','Stkcd']] # dcovar的数据
    "分别输出"
    monthlydataCoVar.to_csv('~/financial_spillover_network/ResData/Risk/CoVarMonthRes.csv',index=False)
    monthlydatadCoVar.to_csv('~/financial_spillover_network/ResData/Risk/dCoVarMonthRes.csv',index=False)
    print(monthlydatacovar)

    "mes的数据月均值计算"
    stocklist = list(set(datames.Stkcd.tolist()))  # 不重复的list
    for stock in stocklist:
        stock_data = datames[datames['Stkcd']==stock].drop('Stkcd',axis=1)
        stock_data_group = stock_data.groupby('monthdate').mean() # 按照monthdate分组，并且求每组内，每列的均值
        # 绘制展示（可以不用）
        # stock_data_group.plot()
        # plt.title(f'the risk indicators of {stock} ')
        # plt.show()
        stock_data_group['Stkcd'] = stock
        monthlydatames = pd.concat([monthlydatames,stock_data_group],axis=0)

    monthlydatames = monthlydatames.reset_index()
    monthlydatames.to_csv('~/financial_spillover_network/ResData/Risk/MESMonthRes.csv',index=False)
    print(monthlydatames)

    return None

if __name__ == '__main__':
    # 文件地址
    filename = '~/financial_spillover_network/ResData/Risk/tailrisk_proxies.csv'
    dfrisk = pd.read_csv(filename)
    newdata  = data_process(dfrisk) # 时间设为index
    "测试covar的计算"
    # covardata = CoVaR_invoke(newdata)
    # mesdata = MES_invoke(newdata)
    
    "输出"
    # filecovar = '~/financial_spillover_network/ResData/Risk/CoVarres.csv'
    # filemes = '~/financial_spillover_network/ResData/Risk/MESres.csv'
    # covardata1 = pd.read_csv(filecovar)
    # mesdata1 = pd.read_csv(filemes)
    # risk_monthly_process(covardata1,mesdata1)
    # 测试输出的数据
    # print(pd.read_csv('~/financial_spillover_network/ResData/Risk/MESMontgRes.csv'))
    
    "测试人工智能的代码"
    test = covar_measurement(newdata,0.6)
    # Example usage
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    alpha = 0.05
    weights = np.array([0.2, 0.3, 0.1, 0.25, 0.15])
    CoVaR = covar_measurement.CoVaR2(returns, alpha, weights)
    print(CoVaR)
    