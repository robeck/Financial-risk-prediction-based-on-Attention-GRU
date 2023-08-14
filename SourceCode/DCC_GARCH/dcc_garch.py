'''
@Author: hoazhi chen
@Date: 2022.3
@Targey: 借助Got上的DCC_Garch包，实现DCC——garch模型的构建，用以输出时间序列之间的动态相关性

目前 dcc_garch 函数只是将收益序列的相关系数进行了输出，没有进一步计算 CoVaR的情况
'''
# 我们从Github上导入了DCC_GARCH包 用于计算动态相关系数！
import pandas as pd
import numpy as np
import warnings
# git上的package，注意看参考
from SourceCode.DCC_GARCH.GARCH.GARCH import GARCH
from SourceCode.DCC_GARCH.GARCH.GARCH_loss import garch_loss_gen
from SourceCode.DCC_GARCH.DCC.DCC import DCC
from SourceCode.DCC_GARCH.DCC.DCC_loss import dcc_loss_gen
from SourceCode.DCC_GARCH.DCC.DCC_loss import R_gen
from arch import arch_model  #arch package

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  #忽略告警

jpm = '~/EnergyriskProject/DCC_GARCH/data/JPM.csv'
sp = '~/EnergyriskProject/DCC_GARCH/data/^GSPC.csv'

'''
1）用于读取测试数据，检查此处工作的正确性！ 
'''
def read_data():
    jpdf = pd.read_csv(jpm).set_index('Date')
    spdf = pd.read_csv(sp).set_index('Date')
    # df.show()
    # jpdf['Adj Close'].plot()

    jpdf_return = np.log(jpdf['Adj Close']).diff().dropna()
    # jpdf_return.drop(index=jpdf.index[[0]],inplace=True)
    spdf_return = np.log(spdf['Adj Close']).diff().dropna()
    # spdf_return.drop(index=spdf.index[[0]],inplace=True)

    #这么做的目的是让数据从T开始一直变到0
    #T，T-1，T-2，。。。，0
    # spdf_return = spdf_return.iloc[::-1]
    # jpdf_return = jpdf_return.iloc[::-1]


    return jpdf_return,spdf_return

'''
来自Topciamine的package，注意检查Git上的文档
该方法下形成的模型可能存在问题！

这里这个模型，我们只需要导入两个时间序列即可！
'''
def Topacm_grach_model(jpm,sp):
    '''
    :arg
        jpm: 时间序列参数1
        sp: 时间序列参数2

    :return
        根据package给予的example，我们发现返回的是
        epsilon = return / sigma  #标准化后的结果！！根据文献！这里应该是残差的标准化才堆，那么这个公式是对残差进行标准化嘛？
    '''
    # 两组数据不同的，分别建立模型
    sp_model = GARCH(1,1)
    sp_model.set_loss(garch_loss_gen(1,1))  #设置损失函数
    sp_model.set_max_itr(1)  #设置最大迭代
    # 拟合
    sp_model.fit(sp)
    # 查看拟合后的参数
    sp_parms = sp_model.get_theta()

    jpm_model = GARCH(1,1)
    jpm_model.set_loss(garch_loss_gen(1,1))
    jpm_model.set_max_itr(1)
    # 拟合
    jpm_model.fit(jpm)
    # 查看拟合后的参数
    # jpm_parms = jpm_model.get_theta()

    # 返回两个序列的epsilon： 这个epsilon是啥？我也不知道
    # 现在知道了 sigma = volatili。 epsilon = 标准化后的残差
    sp_sigma = sp_model.sigma(sp)
    jpm_sigma = jpm_model.sigma(jpm)
    
    sp_epsilon = sp/sp_sigma
    jpm_epsilon = jpm/jpm_sigma
    

    return sp_epsilon,jpm_epsilon

'''
1)这里使用arch包进行GARCH模型的实现
2）可以直接输出condition volatility这个值，非常好用
'''
def arch_garch_model(sp,jpm):
    '''
    :arg
        jpm: 输入的时间序列数据集1
        sp: 输入的时间序列数据集2
    :return
        jpm_epsilon: 输入数据集1的 volatility
        sp_epsilon: 输入数据集2的 volatility
    '''
    "正常的GARCH(1,1)模型"
    # garch_sp = arch_model(sp,vol='GARCH',p=1,o=0,q=1,dist='normal')
    # garch_jpm = arch_model(jpm,vol='GARCH',p=1,o=0,q=1,dist='normal')

    "T-GARCH 模型：p,q,o =1, power=1.0"
    garch_sp = arch_model(sp,p=1,o=1,q=1,power=1.0)
    garch_jpm = arch_model(jpm,p=1,o=1,q=1,power=1.0)

    res_sp = garch_sp.fit(update_freq=5)
    res_jpm = garch_jpm.fit(update_freq=5)
    # .fit（）输出的结果可以调用的值有
    '''
    :Attribute
        aic
        bic
        condition_volatility : 估计的条件波动率！
        resid
        tvalue
        ..
        ..
    '''
    # 计算出来两组波动率 （就是Topacm 提供的包中计算的sigma）
    # sigma 一般就是模型拟合的结果中的 条件波动率（condition volatility）的平方，这里需要进行一下开根号
    volatility_sp = np.sqrt(res_sp.conditional_volatility)
    volatility_jpm = np.sqrt(res_jpm.conditional_volatility)

    sp_epsilon = sp/volatility_sp
    jpm_epsilon = jpm/volatility_jpm

    # print(sp_epsilon)
    # print(jpm_epsilon)

    return sp_epsilon,jpm_epsilon


'''
DCC模型，用于构建动态相关性！输出相关性系数！
'''
def dcc_model(sp_epsilon,jpm_epsilon):
    '''
    :arg
        sp epsilon：标准化后的 i 收益序列
        jpm epsilon 标准化后的 j 收益序列
        
    :return
        rlist: dcc生成的相关系数序列
    '''
    epsilon = np.array([sp_epsilon,jpm_epsilon]) #组合两组收益序列（标准化后的
    dcc_model = DCC()
    dcc_model.set_loss(dcc_loss_gen())
    # 拟合模型
    dcc_model.fit(epsilon)
    # 计算动态相关系数矩阵！Rlist ——>来自于dcc_loss中的R_gen()函数
    ab = dcc_model.get_ab()
    "检查ab值"
    # print([np.round(x,5) for x in ab])

    Rlist = R_gen(epsilon,ab)  #相关系数的矩阵，需要进一步处理

    """
    这里的意义是什么？
    因为代码中，计算的R是针对时间序列数据的每一对数据的，因此这里有一个较长维度的相关性矩阵。
    处理方法：
            1）提取相关性的值
            2）计算一个时间序列的均值
    """
    rlist = []
    for R in Rlist: # 这个list的长度和数据长度是一样的
        # print(R)
        # print(R[0][1])
        rlist.append(R[0][1])

    return rlist #这个返回是输入的时间序列长度的相关性系数序列

'''
是一个main函数，汇统所有模型，输出需要的参数即：dcc的值
'''
def dcc_relation(data1,data2):
    data1_epsilon,data2_epsilon = arch_garch_model(data1,data2)
    value_list = dcc_model(data1_epsilon,data2_epsilon)

    return np.mean(value_list)

if __name__ == '__main__':


    # 1）读取数据，2）使用Topacm的模型计算Volatility，3）dcc模型计算相关系数
    jpdf_return,spdf_return = read_data()
    # print(jpdf_return)
    # print(spdf_return)
    "作者自带的garch模型，存在问题"
    # sp_epsilon,jpm_epsilon = Topacm_grach_model(jpdf_return,spdf_return)
    # value = dcc_model(sp_epsilon,jpm_epsilon)
    # print(np.mean(value))
    "我们自己使用现有的garch模型，应该是对的"
    # sp_epsilon,jpm_epsilon = arch_garch_model(spdf_return,jpdf_return)
    # value = dcc_model(sp_epsilon,jpm_epsilon)
    # print(np.mean(value))

    print(dcc_relation(spdf_return,jpdf_return))
    # print(dcc_relation(jpdf_return,spdf_return))


    # arch_garch_model_forecast(jpdf_return,spdf_return)