""":比较模型



"""
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import *  # 为下面的在错误进行替换，规避不必要的错误提示
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint  # 模型回调，用于进行简单的模型调优
from tensorflow.python.keras.metrics import RootMeanSquaredError  # 模型回调的参数
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV  # 超参寻优

from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score,RocCurveDisplay # 使用sklearn计算准确率
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score # 误差计算
from math import sqrt # 平方
# from tensorflow.keras.layers import Dense, Dropout, LSTM # connot find reference in keras _init_：但其不影响程序的运行

# 测试引入attention机制
from SourceCode.ML.attention_utils import Attention


class Comparable_model():

    def __init__(self, trainx, trainy, valx, valy, testx, testy, features, sc):
        self.Trainx = trainx
        self.Trainy = trainy
        self.Valx = valx
        self.Valy = valy
        self.Testx = testx
        self.Testy = testy
        self.Features = features
        self.sc = sc

    def __str__(self):
        return f'there are two different LSTM model with control num: [1,2]'

    '''
    LSTM 模型1，常规的预测模型，参数手动设定
    '''
    def LSTM_Model(self):
        print(f'you choose the Attention LSTM model')
        ##########################################
        "sequential 结构的模型"

        ##########################################
        "复合结构的模型"
        model_input = Input(shape=(self.Trainx.shape[1], self.Trainx.shape[2]))
        x = LSTM(64, return_sequences=True)(model_input)
        x = Attention(units=32)(x)
        x = Dense(1)(x)
        mymodel = Model(model_input, x)
        #########################################

        mymodel.summary()  # 检查模型的结构
        mymodel.compile(loss='mae', optimizer='adam')
        "这里其实是使用测试集来分析误差"
        history = mymodel.fit(self.Trainx, self.Trainy, batch_size=64, epochs=50,
                              validation_data=(self.Valx, self.Valy),
                              validation_freq=1)
        ''':return
        mymodel : 自己训练好的模型
        history : history中会存储loss，val_loss的数据
        '''
        return mymodel, history

    '''
    LSTM 模型2: 在常规模型的基础上进行了参数的GridSearch cross validation
    '''

    def GRU_Model(self):
        print(f'you choose the Attention LSTM model')
        ##########################################
        "sequential 结构的模型"

        ##########################################
        "复合结构的模型"
        model_input = Input(shape=(self.Trainx.shape[1], self.Trainx.shape[2]))
        x = Dense(64,activation='relu')(model_input)
        x = GRU(64, return_sequences=True)(x)
        x = Dense(32)(x)
        x = Dense(1)(x)
        mymodel = Model(model_input, x)
        #########################################

        mymodel.summary()  # 检查模型的结构
        mymodel.compile(loss='mae', optimizer='adam')
        "这里其实是使用测试集来分析误差"
        history = mymodel.fit(self.Trainx, self.Trainy, batch_size=64, epochs=50,
                              validation_data=(self.Valx, self.Valy),
                              validation_freq=1)
        ''':return
        mymodel : 自己训练好的模型
        history : history中会存储loss，val_loss的数据
        '''
        return mymodel, history

    ''':prediction
    1) 计算prediction的值
    2）输出prediction和real值
    PS：这里的工作就是进行预测，并且将预测结果逆向化回来
    '''

    def prediction(self, model, n):
        "1. 预测数据，张量转换，去归一化"
        predict_price = model.predict(self.Testx)
        features = self.Features + n
        '''
        2. 逆缩放
        一般会报错 non-broadcastable output operand with shape (1630,1) doesn't match the broadcast shape (1630,8)
        因为，我们在缩放数据的时候有：features个列，但是现在只有1列！！！！！
        所以，我们必须改变数据形状来进行逆缩放

        核心：因为sc做缩放的时候特征是 n 个，这里如果逆缩放特征个数少于 n 是不行的！
        '''
        prediction_copy_array = np.repeat(predict_price, features)  # 复制feature个列！用于逆缩放
        # 这里面还需要reshape，因为虽然我们repeat了数据，但是并没有改变数据的结构，还需要reshape一下
        predict_price = self.sc.inverse_transform(np.reshape(prediction_copy_array, (len(predict_price), features)))[:,
                        0]  # 逆缩放的结果提取第一列就是我们要的预测值
        real_price_copy = np.repeat(self.Testy, features)
        real_price = self.sc.inverse_transform(np.reshape(real_price_copy, (len(self.Testy), features)))[:, 0]

        return predict_price, real_price



def SVR_model(data):
    '''
    :param datalist: 数据要进行标准化！
    :return:
    '''
    # 读取的数据处理
    dataX,dataY = np.array(data.iloc[:,1:]),np.array(data.iloc[:,0]).ravel()
    train_X, test_X, train_y, test_y = train_test_split(dataX, dataY, random_state=0, test_size=0.3)

    "建模拟合（没有调参）"
    clf = SVR(kernel='linear',gamma='auto')
    clf.fit(train_X,train_y)

    "预测+绘制"
    prediction = clf.predict(test_X) # 预测结果

    "结果评价"
    # calculate MSE 均方误差
    mse = mean_squared_error(test_y, prediction)
    # calculate RMSE 均方根误差
    rmse = sqrt(mean_squared_error(test_y, prediction))
    # calculate MAE 平均绝对误差
    mae = mean_absolute_error(test_y, prediction)
    # calculate R square
    r_square = r2_score(test_y, prediction)
    print('均方误差 mse : %.6f' % mse)
    print('均方根误差 rmse : %.6f' % rmse)
    print('平均绝对误差 mae : %.6f' % mae)
    print('R_square: %.6f' % r_square)


    return clf

