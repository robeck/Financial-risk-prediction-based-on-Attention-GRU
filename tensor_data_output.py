
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from SourceCode.ML.LSTM import Manual_LSTM
from SourceCode.ML.Comparable_model import Comparable_model,SVR_model
from SourceCode.ML.Model_eva import calculate_mape # 计算mape

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score # 误差计算
from math import sqrt # 平方


# sc = MinMaxScaler(feature_range=(0,1)) # 0-1标准化 z-scores
sc = StandardScaler() # 标准化


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
    train_length,val_length,test_length = int(np.ceil(rows * 0.6)),int(np.ceil(rows * 0.8)),int(np.ceil(rows))
    train_set = df.iloc[0:train_length,:].values
    val_set = df.iloc[train_length:val_length,:].values
    test_set = df.iloc[val_length:,:].values

    "一个数据标准换的工作（可以不做，可以在模型中增加）"
    train_set = sc.fit_transform(train_set)
    val_set = sc.fit_transform(val_set)
    test_set = sc.fit_transform(test_set)

    "制作张量结构的数据Tensor data"
    window_size = 15 # 20日的 time_step
    X_train, Y_train, X_val, Y_val, X_test, Y_test = [],[],[],[],[],[]
    # train 数据集部分
    for i in range(0,len(train_set)-window_size):
        x_tensor = train_set[i:i+window_size,1:]
        X_train.append(x_tensor)
        y_tensor = train_set[i+window_size,0]
        Y_train.append(y_tensor)
    X_train,Y_train = np.array(X_train),np.array(Y_train)
    Y_train = np.reshape(Y_train,(Y_train.shape[0],1))
    # val 数据集部分
    for i in range(0,len(val_set)-window_size):
        x_tensor = val_set[i:i+window_size,1:]
        X_val.append(x_tensor)
        y_tensor = val_set[i+window_size,0]
        Y_val.append(y_tensor)
    X_val,Y_val = np.array(X_val),np.array(Y_val)
    Y_val = np.reshape(Y_val,(Y_val.shape[0],1))
    # test 数据集部分
    for i in range(0,len(test_set)-window_size):
        x_tensor = test_set[i:i+window_size,1:]
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
def dftest_to_tf(df):
    df.dropna(axis=0,inplace=True) # 一定要去除nan值，否则会出现神经元坏死的情况
    rows = df.shape[0]

    "划分测试和训练数据，不使用sklearn的自划分方法 train_test_split"
    test_set = df.iloc[:,:].values

    "一个数据标准换的工作（可以不做，可以在模型中增加）"
    test_set = sc.fit_transform(test_set)

    "制作张量结构的数据Tensor data"
    window_size = 10 # 20日的 time_step
    X_test, Y_test = [],[]
    # test 数据集部分
    for i in range(0,len(test_set)-window_size):
        x_tensor = test_set[i:i+window_size,1:]
        X_test.append(x_tensor)
        y_tensor = test_set[i+window_size,0]
        Y_test.append(y_tensor)
    X_test,Y_test = np.array(X_test),np.array(Y_test)
    Y_test = np.reshape(Y_test,(Y_test.shape[0],1))

    features = X_test.shape[1] #特征的个数，也是列的个数
    Features = df.shape[1]
    n = Features - features # 训练模型的特征和总特征差 为 n 要用作预测结果逆归一化的补充
    return X_test,Y_test,features,n

