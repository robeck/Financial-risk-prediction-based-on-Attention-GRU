'''
@Author: haozhic
@Date: 2022-07
@Target: 实现能源公司的风险预警研究
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from SourceCode.ML.LSTM import Manual_LSTM

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score # 误差计算
from math import sqrt # 平方

sc = MinMaxScaler(feature_range=(0,1)) # 0-1标准化 z-scores
from tqdm import tqdm


'''真正的主函数
1）调用模型用的
'''
def main():
    filename = '~/financial_spillover_network/Data/Energycompanydata/Energy_companies_trade_data.csv'
    df = pd.read_csv(filename)
    stocklist = list(set(df.Stkcd.tolist()))

    df = df[['Clsprc','Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Dnshrtrd',
        'Dsmvosd', 'PreClosePrice', 'ChangeRatio']] #截取需要的数据部分
    # df = df[['Clsprc','Stkcd','Trddt']] # 只用收益率序列

    print(df.head())


    for stockid in stocklist:
        stock_data = df[df.Stkcd==stockid].set_index('Trddt')
        stock_data.drop(['Stkcd'],axis=1,inplace=True)
        if stock_data.shape[0] > 200:
            X_train,Y_train,X_val,Y_val,X_test,Y_test,features,n = df_to_tf(stock_data)
            print(X_train)
            model_invoke((X_train,Y_train,X_val,Y_val,X_test,Y_test),features,n)
        else:
            continue


    return None


'''一个全局可调用的dataframe转tensor的函数
Input:
    df：结构应该是没有Stkcd，Trddt做为index的一个daatframe
return:
    考虑输出全套数据
    X_train
    Y_train
    X_val
    Y_val
    X_test
    Y_test
'''
def df_to_tf(df):
    df.dropna(axis=0,inplace=True) # 一定要去除nan值，否则会出现神经元坏死的情况
    rows = df.shape[0]

    "划分测试和训练数据，不使用sklearn的自划分方法 train_test_split"
    train_length,val_length,test_length = int(np.ceil(rows * 0.7)),int(np.ceil(rows * 0.9)),int(np.ceil(rows))
    train_set = df.iloc[0:train_length,:].values
    val_set = df.iloc[train_length:val_length,:].values
    test_set = df.iloc[val_length:,:].values

    "一个数据标准换的工作（可以不做，可以在模型中增加）"
    train_set = sc.fit_transform(train_set)
    val_set = sc.fit_transform(val_set)
    test_set = sc.fit_transform(test_set)

    "制作张量结构的数据Tensor data"
    window_size = 10 # 20日的 time_step
    X_train, Y_train, X_val, Y_val, X_test, Y_test = [],[],[],[],[],[]
    # train 数据集部分
    for i in range(0,len(train_set)-window_size):
        x_tensor = train_set[i:i+window_size,0:] # 行 [i:i+window_size] 为训练的X。[1：] 表示的是 学习的特征，如何这里是0，表示只有一个特征即用来学，也用于预测
        X_train.append(x_tensor)
        y_tensor = train_set[i+window_size,0]
        Y_train.append(y_tensor)
    X_train,Y_train = np.array(X_train),np.array(Y_train)
    Y_train = np.reshape(Y_train,(Y_train.shape[0],1))
    # val 数据集部分
    for i in range(0,len(val_set)-window_size):
        x_tensor = val_set[i:i+window_size,0:]
        X_val.append(x_tensor)
        y_tensor = val_set[i+window_size,0]
        Y_val.append(y_tensor)
    X_val,Y_val = np.array(X_val),np.array(Y_val)
    Y_val = np.reshape(Y_val,(Y_val.shape[0],1))
    # test 数据集部分
    for i in range(0,len(test_set)-window_size):
        x_tensor = test_set[i:i+window_size,0:]
        X_test.append(x_tensor)
        y_tensor = test_set[i+window_size,0]
        Y_test.append(y_tensor)
    X_test,Y_test = np.array(X_test),np.array(Y_test)
    Y_test = np.reshape(Y_test,(Y_test.shape[0],1))

    features = X_train.shape[1] #特征的个数，也是列的个数
    Features = df.shape[1]
    n = Features - features # 训练模型的特征和总特征差 为 n 要用作预测结果逆归一化的补充
    
    return X_train,Y_train,X_val,Y_val,X_test,Y_test,features,n

''':数据预处理
1）数据处理过后进行保存
2）保存后的数据单独进行读取即可
'''
def data_preprocess_main():
    stocklist_filename = '~/financial_spillover_network/Data/Energycompanydata/Energy_companies_wind.csv'
    stocktrade_filename = '~/financial_spillover_network/Data/Energycompanydata/StockDaily.csv'
    stocklist = [int(stockcd[0:6]) for stockcd in pd.read_csv(stocklist_filename).Stkcd.tolist()]
    print(stocklist)

    trade_data = pd.read_csv(stocktrade_filename)
    print(trade_data)

    reconstructed_data = pd.DataFrame()
    for stockid in tqdm(stocklist):
        stockdata = trade_data[trade_data['Stkcd']==stockid]
        reconstructed_data = pd.concat([reconstructed_data,stockdata],axis=0)

    reconstructed_data = reconstructed_data.reset_index().drop('index',axis=1)
    reconstructed_data.to_csv('~/financial_spillover_network/Data/Energycompanydata/Energy_companies_trade_data.csv',index=False)
    print(reconstructed_data)
    return None


'''模型导入，输出预测结果的函数
input：
    parameters：包含了全部train，val和test数据
    features: 实际用到的特征个数
    n：总特征和实际特征差
return:
    None
'''
def model_invoke(Parameters,features,n):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        Parameters[0],Parameters[1],Parameters[2],Parameters[3],Parameters[4],Parameters[5]
    "LSTM model"
    lstm_instance = Manual_LSTM(X_train, Y_train, X_val, Y_val, X_test, Y_test,features,sc)
    "下面的LSTM选一个即可"
    # model,history = lstm_instance.LSTM_Model1() # 基础模型
    # model, history = lstm_instance.LSTM_Model2() # 使用交叉验证
    model, history = lstm_instance.LSTM_Model3() # GRU模型

    "给到loss,val_loss的绘图"
    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='test')
    plt.title('LSTM_train_test_lostt',fontsize=20)
    plt.show()

    "给与一个预测结果"
    predic_price,real_price = lstm_instance.prediction(model,n)
    plt.plot(real_price,color='red')
    plt.plot(predic_price,color='blue')
    plt.show()

    "结果评价"
    # calculate MSE 均方误差
    mse = mean_squared_error(real_price, predic_price)
    # calculate RMSE 均方根误差
    rmse = sqrt(mean_squared_error(real_price, predic_price))
    # calculate MAE 平均绝对误差
    mae = mean_absolute_error(real_price, predic_price)
    # calculate R square
    r_square = r2_score(real_price, predic_price)
    print('均方误差 mse : %.6f' % mse)
    print('均方根误差 rmse : %.6f' % rmse)
    print('平均绝对误差 mae : %.6f' % mae)
    print('R_square: %.6f' % r_square)

    return None



if __name__ == '__main__':
    # 非完整数据集处理部分
    # data_preprocess_main()

    "主程序运行"
    main()
    # file = "~/ListedCompany_risk/Data/StockRiskWarning.csv"
    # df = pd.read_csv(file).set_index('Date')
    # print(df)
