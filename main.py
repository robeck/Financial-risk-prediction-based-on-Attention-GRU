# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
'''
@Author: haozhic
@Date: 2022-07
@Target: 实现能源公司的风险预警研究
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from SourceCode.ML.LSTM import Manual_LSTM
from SourceCode.ML.Comparable_model import Comparable_model,SVR_model
from SourceCode.ML.Model_eva import calculate_mape # 计算mape
import tensor_data_output

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score # 误差计算
from math import sqrt # 平方

from tqdm import tqdm


# sc = MinMaxScaler(feature_range=(0,1)) # 0-1标准化 z-scores
sc = StandardScaler() # 标准化


'''真正的主函数
1）调用模型用的
'''
def main(method):
    # 初始化参数
    # risknamelist = ['CoVar', 'dCoVar', 'MES', 'Beta', 'Vol', 'Turn', 'Cor', 'Illiq']
    risknamelist = ['CoVar', 'dCoVar', 'Beta', 'Vol', 'Turn', 'Cor', 'Illiq'] # 没有mes

    if method==1:
        "方案1：先对每个公司划分数据集，再分别汇总训练，验证和测试，数据集构建成tensor"
        "df to tf 参数"
        trainpos, valpos, stamps, feapos = 0.6,0.8,15,0
        "循环读取每一个风险数据"
        for i, riskname in enumerate(risknamelist):
            df = pd.read_csv(f'~/financial_spillover_network/ResData/Risk/{riskname}Month_mergetopo_Res.csv')
            stocklist = list(set(df.Stkcd.tolist()))# 股票序列

            trainstocklist = stocklist[0:-3] # 测试下一个想法，前面一大部分股票进行训练
            X_train_tesor,Y_train_tesor,X_val_tesor,Y_val_tesor,X_test_tesor,Y_test_tesor = [],[],[],[],[],[] # tensor合并
            features,n = 0,0
            "循环每一个公司"
            for stockid in trainstocklist: # 注意，需要调换
                stock_data = df[df.Stkcd==stockid].set_index('monthdate') # 日期边索引
                stock_data.drop(['Stkcd'],axis=1,inplace=True) # 提出不要的stkcd号

                if stock_data.shape[0] > 45:
                    X_train,Y_train,X_val,Y_val,X_test,Y_test,features,n = df_to_tf(stock_data,trainpos, valpos, stamps, feapos)
                    ######################### "尝试汇总tensor来训练"
                    X_train_tesor.append(X_train)
                    Y_train_tesor.append(Y_train)
                    X_val_tesor.append(X_val)
                    Y_val_tesor.append(Y_val)
                    X_test_tesor.append(X_test)
                    Y_test_tesor.append(Y_test)
                    features = features
                    n = n
                    ########################
                    print(X_val.shape)
                    # model_invoke((X_train,Y_train,X_val,Y_val,X_test,Y_test),features,n)
                else:
                    continue

            X_train, X_val, X_test = tf.concat(X_train_tesor, axis=0), tf.concat(X_val_tesor, axis=0), tf.concat(X_test_tesor, axis=0)
            Y_train, Y_val, Y_test = tf.concat(Y_train_tesor, axis=0), tf.concat(Y_val_tesor, axis=0), tf.concat(Y_test_tesor, axis=0)

            "整体预测"
            model_instance,model,n = model_invoke((X_train, Y_train, X_val, Y_val, X_test, Y_test), features, n)

            ''':预测方案1
            这里我们做一个很有意思的事情，我们训练的模型是使用组合多个股票的风险数据的张量。
            预测这里，我们分别对每一个股票进行结果张量的预测，看看怎么样！
            '''
            # for i,x_test in enumerate(X_test_tesor):
            #     y_test = Y_test_tesor[i]
            #     model_predict(model_instance,model,n,x_test,y_test)

            ''':预测方案2
            我们这里对选几只股票进行预测，因此需要有个单独构建tensor的部分
            '''
            resstocklist = stocklist[-3:]
            "循环每一个公司"
            for stockid in resstocklist:
                stock_data = df[df.Stkcd == stockid].set_index('monthdate')  # 日期边索引
                stock_data.drop(['Stkcd'], axis=1, inplace=True)  # 提出不要的stkcd号
                X_test_tesor, Y_test_tesor = [], []
                if stock_data.shape[0] > 45:
                    X_test, Y_test, features, n = dftest_to_tf(stock_data,stamps,feapos)
                    ######################### "尝试汇总tensor来训练"
                    X_test_tesor.append(X_test)
                    Y_test_tesor.append(Y_test)
                    features = features
                    n = n
                    ########################
                    "内部一个个来，进行预测"
                    model_predict(model_instance, model, n, X_test, Y_test)
                else:
                    continue
            "外部，汇总张量后预测"
            X_test,Y_test = tf.concat(X_test_tesor, axis=0),tf.concat(Y_test_tesor, axis=0)
            model_predict(model_instance, model, n, X_test, Y_test)

            "比较模型的整体预测"
            df = df.set_index('monthdate').drop(['Stkcd'],axis=1)
            print(df.head())
            # comparable(df,(X_train, Y_train, X_val, Y_val, X_test, Y_test), features, n)
    elif method == 2:
        "方案2：一起划分，训练和预测"
        "df to tf 参数"
        trainpos, valpos, stamps, feapos = 0.8,0.9,60,0
        "循环读取每一个风险数据"
        for i, riskname in enumerate(risknamelist):
            df = pd.read_csv(f'~/financial_spillover_network/ResData/Risk/{riskname}Month_mergetopo_Res.csv')
            df = df.set_index('monthdate').drop(['Stkcd'],axis=1)
            # df = df.sort_index() # 按照索引排序！
            print(df) # 检查样式
            "整体预测"
            X_train, Y_train, X_val, Y_val, X_test, Y_test, features, n = df_to_tf(df,trainpos, valpos, stamps, feapos)
            model_instance, model, n = model_invoke((X_train, Y_train, X_val, Y_val, X_test, Y_test), features, n)
            "比较模型"
            # comparable(df,(X_train, Y_train, X_val, Y_val, X_test, Y_test), features, n)
    else:
        pass

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
    
注意三个参数
1）训练，测试：数据集的位置
2）timestamp
'''
def df_to_tf(df,trainpos,valpos,stamps,feapos):
    '''
    :param df: 数据集
    :param trainpos: 训练数据，验证数据比例
    :param valpos: 验证数据，测试数据比例
    :param stamps: 时间戳
    :param feapos: 特征所在位置
    :return:
    '''
    df.dropna(axis=0,inplace=True) # 一定要去除nan值，否则会出现神经元坏死的情况
    rows = df.shape[0]

    "划分测试和训练数据，不使用sklearn的自划分方法 train_test_split"
    train_length,val_length,test_length = int(np.ceil(rows * trainpos)),int(np.ceil(rows * valpos)),int(np.ceil(rows))
    train_set = df.iloc[0:train_length,:].values
    val_set = df.iloc[train_length:val_length,:].values
    test_set = df.iloc[val_length:,:].values

    "一个数据标准换的工作（可以不做，可以在模型中增加）"
    train_set = sc.fit_transform(train_set)
    val_set = sc.fit_transform(val_set)
    test_set = sc.fit_transform(test_set)

    "制作张量结构的数据Tensor data"
    window_size = stamps # 20日的 time_step
    X_train, Y_train, X_val, Y_val, X_test, Y_test = [],[],[],[],[],[]
    # train 数据集部分
    for i in range(0,len(train_set)-window_size):
        x_tensor = train_set[i:i+window_size,feapos:]
        X_train.append(x_tensor)
        y_tensor = train_set[i+window_size,0]
        Y_train.append(y_tensor)
    X_train,Y_train = np.array(X_train),np.array(Y_train)
    Y_train = np.reshape(Y_train,(Y_train.shape[0],1))
    # val 数据集部分
    for i in range(0,len(val_set)-window_size):
        x_tensor = val_set[i:i+window_size,feapos:]
        X_val.append(x_tensor)
        y_tensor = val_set[i+window_size,0]
        Y_val.append(y_tensor)
    X_val,Y_val = np.array(X_val),np.array(Y_val)
    Y_val = np.reshape(Y_val,(Y_val.shape[0],1))
    # test 数据集部分
    for i in range(0,len(test_set)-window_size):
        x_tensor = test_set[i:i+window_size,feapos:]
        X_test.append(x_tensor)
        y_tensor = test_set[i+window_size,0]
        Y_test.append(y_tensor)
    X_test,Y_test = np.array(X_test),np.array(Y_test)
    Y_test = np.reshape(Y_test,(Y_test.shape[0],1))

    features = X_train.shape[1] #特征的个数，也是列的个数
    Features = df.shape[1]
    n = Features - features # 训练模型的特征和总特征差 为 n 要用作预测结果逆归一化的补充
    return X_train,Y_train,X_val,Y_val,X_test,Y_test,features,n


"所有数据都作为test数据"
def dftest_to_tf(df,stamps,feapos):
    df.dropna(axis=0,inplace=True) # 一定要去除nan值，否则会出现神经元坏死的情况
    rows = df.shape[0]

    "划分测试和训练数据，不使用sklearn的自划分方法 train_test_split"
    test_set = df.iloc[:,:].values

    "一个数据标准换的工作（可以不做，可以在模型中增加）"
    test_set = sc.fit_transform(test_set)

    "制作张量结构的数据Tensor data"
    window_size = stamps # 20日的 time_step
    X_test, Y_test = [],[]
    # test 数据集部分
    for i in range(0,len(test_set)-window_size):
        x_tensor = test_set[i:i+window_size,feapos:]
        X_test.append(x_tensor)
        y_tensor = test_set[i+window_size,0]
        Y_test.append(y_tensor)
    X_test,Y_test = np.array(X_test),np.array(Y_test)
    Y_test = np.reshape(Y_test,(Y_test.shape[0],1))

    features = X_test.shape[1] #特征的个数，也是列的个数
    Features = df.shape[1]
    n = Features - features # 训练模型的特征和总特征差 为 n 要用作预测结果逆归一化的补充
    return X_test,Y_test,features,n





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
    # model,history = lstm_instance.LSTM_Model() # 基础模型
    # model,history = lstm_instance.LSTM_Model1() # 基础模型
    # model, history = lstm_instance.LSTM_Model2() # 使用交叉验证
    # model, history = lstm_instance.LSTM_Model3() # GRU模型
    model, history = lstm_instance.LSTM_Model4()  # GRU模型
    "给到loss,val_loss的绘图"
    print('RMSE: ' + str(np.mean(history.history['loss'])))
    print('Val RMSE: ' + str(np.mean(history.history['val_loss'])))
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
    # calculate mape
    mape = calculate_mape(real_price, predic_price)
    print('均方误差 mse : %.6f' % mse)
    print('均方根误差 rmse : %.6f' % rmse)
    print('平均绝对误差 mae : %.6f' % mae)
    print('平均绝对百分比误差 mape : %.6f' % mape)
    print('R_square: %.6f' % r_square)

    "可以返回参数或者none"
    # return None
    return lstm_instance,model,n

"一个单独预测的部分"
'''传入参数：
    model_instance
'''
def model_predict(model_instance,model,n,testx,testy):
    "给与一个预测结果"
    predic_price,real_price = model_instance.prediction_exter(model,n,testx,testy)

    "结果评价"
    # calculate MSE 均方误差
    mse = mean_squared_error(real_price, predic_price)
    # calculate RMSE 均方根误差
    rmse = sqrt(mean_squared_error(real_price, predic_price))
    # calculate MAE 平均绝对误差
    mae = mean_absolute_error(real_price, predic_price)
    # calculate R square
    r_square = r2_score(real_price, predic_price)
    # calculate mape
    mape = calculate_mape(real_price, predic_price)
    print('均方误差 mse : %.6f' % mse)
    print('均方根误差 rmse : %.6f' % rmse)
    print('平均绝对误差 mae : %.6f' % mae)
    print('平均绝对百分比误差 mape : %.6f' % mape)
    print('R_square: %.6f' % r_square)

    return rmse,mae,mape

''':一些比较模型
1）LSTM
2）GRU
3）SVR
'''
def comparable(data,Parameters,features,n):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        Parameters[0], Parameters[1], Parameters[2], Parameters[3], Parameters[4], Parameters[5]
    "comparable model"
    lstm_instance = Comparable_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, features, sc)
    "LSTM即可"
    model,history = lstm_instance.LSTM_Model() # LSTM
    "给到loss,val_loss的绘图"
    print('RMSE: ' + str(np.mean(history.history['loss'])))
    print('Val RMSE: ' + str(np.mean(history.history['val_loss'])))
    "给与一个预测结果"
    predic_price, real_price = lstm_instance.prediction(model, n)
    "结果评价"
    # calculate MSE 均方误差
    mse = mean_squared_error(real_price, predic_price)
    # calculate RMSE 均方根误差
    rmse = sqrt(mean_squared_error(real_price, predic_price))
    # calculate MAE 平均绝对误差
    mae = mean_absolute_error(real_price, predic_price)
    # calculate R square
    r_square = r2_score(real_price, predic_price)
    # calculate mape
    mape = calculate_mape(real_price, predic_price)
    print('均方误差 mse : %.6f' % mse)
    print('均方根误差 rmse : %.6f' % rmse)
    print('平均绝对误差 mae : %.6f' % mae)
    print('平均绝对百分比误差 mape : %.6f' % mape)
    print('R_square: %.6f' % r_square)

    "GRU"
    model, history = lstm_instance.GRU_Model()  # GRU
    "给到loss,val_loss的绘图"
    print('RMSE: ' + str(np.mean(history.history['loss'])))
    print('Val RMSE: ' + str(np.mean(history.history['val_loss'])))
    "给与一个预测结果"
    predic_price, real_price = lstm_instance.prediction(model, n)
    "结果评价"
    # calculate MSE 均方误差
    mse = mean_squared_error(real_price, predic_price)
    # calculate RMSE 均方根误差
    rmse = sqrt(mean_squared_error(real_price, predic_price))
    # calculate MAE 平均绝对误差
    mae = mean_absolute_error(real_price, predic_price)
    # calculate R square
    r_square = r2_score(real_price, predic_price)
    # calculate mape
    mape = calculate_mape(real_price, predic_price)
    print('均方误差 mse : %.6f' % mse)
    print('均方根误差 rmse : %.6f' % rmse)
    print('平均绝对误差 mae : %.6f' % mae)
    print('平均绝对百分比误差 mape : %.6f' % mape)
    print('R_square: %.6f' % r_square)

    "SVR"
    SVR_model(data)


    return None


if __name__ == '__main__':
    # 非完整数据集处理部分
    # data_preprocess_main()

    "主程序运行"
    # method = 1
    method = 2
    main(method)
    # file = "~/ListedCompany_risk/Data/StockRiskWarning.csv"
    # df = pd.read_csv(file).set_index('Date')
    # print(df)
