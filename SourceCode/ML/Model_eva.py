import numpy as np

def calculate_mape(y_true, y_pred):
    """
    计算平均绝对百分比误差（MAPE）

    参数：
        - y_true：实际值
        - y_pred：预测值

    返回值：
        - mape：MAPE值
    """
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape