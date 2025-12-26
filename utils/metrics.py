"""
评价指标模块
实现 MSE, RMSE, MAE, R-Squared 等指标
"""
import numpy as np


def mse(y_true, y_pred):
    """
    均方误差 (Mean Squared Error)
    
    公式: MSE = (1/n) * Σ(y_true - y_pred)^2
    
    参数:
        y_true: numpy array, 真实值
        y_pred: numpy array, 预测值
    
    返回:
        float, MSE 值
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    均方根误差 (Root Mean Squared Error)
    
    公式: RMSE = sqrt(MSE)
    
    参数:
        y_true: numpy array, 真实值
        y_pred: numpy array, 预测值
    
    返回:
        float, RMSE 值
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """
    平均绝对误差 (Mean Absolute Error)
    
    公式: MAE = (1/n) * Σ|y_true - y_pred|
    
    参数:
        y_true: numpy array, 真实值
        y_pred: numpy array, 预测值
    
    返回:
        float, MAE 值
    """
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    """
    决定系数 (R-Squared)
    
    衡量模型对数据的拟合程度，取值范围 (-∞, 1]
    R^2 = 1 表示完美拟合
    R^2 = 0 表示模型预测等同于预测平均值
    R^2 < 0 表示模型比预测平均值还差
    
    公式:
        SS_res = Σ(y_true - y_pred)^2  # 残差平方和
        SS_tot = Σ(y_true - y_mean)^2  # 总平方和
        R^2 = 1 - SS_res / SS_tot
    
    参数:
        y_true: numpy array, 真实值
        y_pred: numpy array, 预测值
    
    返回:
        float, R^2 值
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # 防止除零
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def mape(y_true, y_pred, epsilon=1e-10):
    """
    平均绝对百分比误差 (Mean Absolute Percentage Error)
    
    公式: MAPE = (100/n) * Σ|（y_true - y_pred) / y_true|
    
    参数:
        y_true: numpy array, 真实值
        y_pred: numpy array, 预测值
        epsilon: float, 防止除零的小常数
    
    返回:
        float, MAPE 值 (百分比)
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def compute_all_metrics(y_true, y_pred):
    """
    计算所有评价指标
    
    参数:
        y_true: numpy array, 真实值
        y_pred: numpy array, 预测值
    
    返回:
        dict, 包含所有指标的字典
    """
    return {
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'R2': r_squared(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }


def print_metrics(metrics_dict, model_name="Model"):
    """
    打印评价指标
    
    参数:
        metrics_dict: dict, 指标字典
        model_name: str, 模型名称
    """
    print(f"\n{'='*50}")
    print(f"{model_name} - 评价指标:")
    print(f"{'='*50}")
    print(f"  MSE:  {metrics_dict['MSE']:.6f}")
    print(f"  RMSE: {metrics_dict['RMSE']:.6f}")
    print(f"  MAE:  {metrics_dict['MAE']:.6f}")
    print(f"  R²:   {metrics_dict['R2']:.6f}")
    print(f"  MAPE: {metrics_dict['MAPE']:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    # 测试评价指标
    print("测试评价指标:")
    
    # 生成测试数据
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.3, 4.9])
    
    print(f"\n真实值: {y_true}")
    print(f"预测值: {y_pred}")
    
    # 计算所有指标
    metrics = compute_all_metrics(y_true, y_pred)
    print_metrics(metrics, "测试模型")
    
    print("\n评价指标测试成功!")
