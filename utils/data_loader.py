"""
数据加载与预处理模块
用于下载和处理 AAPL 股票数据，并生成用于训练的批次数据
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def load_yahoo_stock(ticker='AAPL', start='2010-01-01', end='2023-12-31', use_mock_data=False):
    """
    使用 yfinance 下载股票数据并进行归一化
    
    参数:
        ticker: str, 股票代码，默认为 'AAPL'
        start: str, 开始日期
        end: str, 结束日期
        use_mock_data: bool, 是否使用模拟数据（API 限流时使用）
    
    返回:
        data_scaled: numpy array, 归一化后的收盘价数据 (n_samples,)
        scaler: MinMaxScaler 对象, 用于反归一化
        raw_data: pandas Series, 原始收盘价数据
    """
    if use_mock_data:
        print(f"⚠️ 使用模拟 {ticker} 股票数据（用于演示）...")
        return _generate_mock_stock_data()
    
    print(f"正在下载 {ticker} 股票数据 ({start} 到 {end})...")
    
    try:
        # 使用 yfinance 下载数据
        stock_data = yf.download(ticker, start=start, end=end, progress=False)
        
        # 检查数据是否为空
        if stock_data.empty or len(stock_data) == 0:
            print("\n❌ 数据下载失败（可能是 API 限流或网络问题）")
            print("💡 自动切换到模拟数据模式...\n")
            return _generate_mock_stock_data()
        
        # 提取收盘价列
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        
        # 使用 MinMaxScaler 将数据归一化到 (0, 1) 范围
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(close_prices).flatten()
        
        print(f"✅ 数据下载完成！共 {len(data_scaled)} 个数据点")
        print(f"原始数据范围: [{close_prices.min():.2f}, {close_prices.max():.2f}]")
        
        return data_scaled, scaler, stock_data['Close']
        
    except Exception as e:
        print(f"\n❌ 下载数据时发生错误: {str(e)}")
        print("💡 自动切换到模拟数据模式...\n")
        return _generate_mock_stock_data()


def _generate_mock_stock_data():
    """
    生成模拟的股票数据（用于演示或 API 限流时）
    
    返回:
        data_scaled: numpy array, 归一化后的数据
        scaler: MinMaxScaler 对象
        raw_data: pandas Series, 原始数据
    """
    np.random.seed(42)
    
    # 生成模拟数据：基础趋势 + 随机波动
    n_samples = 3000
    t = np.arange(n_samples)
    
    # 长期上升趋势
    trend = 100 + 0.05 * t
    
    # 周期性波动
    seasonal = 10 * np.sin(2 * np.pi * t / 365)
    
    # 随机噪声
    noise = np.random.randn(n_samples) * 5
    
    # 组合成模拟股价
    mock_prices = trend + seasonal + noise
    mock_prices = mock_prices.reshape(-1, 1)
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(mock_prices).flatten()
    
    # 创建 pandas Series
    dates = pd.date_range(start='2010-01-01', periods=n_samples, freq='D')
    raw_data = pd.Series(mock_prices.flatten(), index=dates)
    
    print(f"✅ 生成模拟数据！共 {len(data_scaled)} 个数据点")
    print(f"原始数据范围: [{mock_prices.min():.2f}, {mock_prices.max():.2f}]")
    
    return data_scaled, scaler, raw_data


def create_sequences(data, seq_len=60):
    """
    使用滑动窗口方法创建序列数据
    
    将时间序列数据转换为监督学习问题：
    给定过去 seq_len 个时间步的数据，预测下一个时间步的值
    
    参数:
        data: numpy array, shape=(n_samples,), 时间序列数据
        seq_len: int, 输入序列长度（回看窗口大小）
    
    返回:
        X: numpy array, shape=(n_sequences, seq_len, 1), 输入序列
        y: numpy array, shape=(n_sequences, 1), 目标值
    
    示例:
        data = [1, 2, 3, 4, 5, 6, 7], seq_len = 3
        X = [[1,2,3], [2,3,4], [3,4,5], [4,5,6]]
        y = [4, 5, 6, 7]
    """
    X, y = [], []
    
    for i in range(len(data) - seq_len):
        # 输入序列: data[i : i+seq_len]
        X.append(data[i : i + seq_len])
        # 目标值: data[i+seq_len]
        y.append(data[i + seq_len])
    
    X = np.array(X)  # shape: (n_sequences, seq_len)
    y = np.array(y)  # shape: (n_sequences,)
    
    # 增加特征维度
    X = np.expand_dims(X, axis=-1)  # shape: (n_sequences, seq_len, 1)
    y = np.expand_dims(y, axis=-1)  # shape: (n_sequences, 1)
    
    return X, y


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    将数据划分为训练集、验证集和测试集
    
    参数:
        X: numpy array, 输入序列
        y: numpy array, 目标值
        train_ratio: float, 训练集比例
        val_ratio: float, 验证集比例
    
    返回:
        X_train, y_train: 训练集
        X_val, y_val: 验证集
        X_test, y_test: 测试集
    """
    n_samples = len(X)
    
    # 计算划分点
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # 划分数据
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def batch_generator(X, y, batch_size=32, shuffle=True):
    """
    批次数据生成器
    
    参数:
        X: numpy array, shape=(n_samples, seq_len, input_size)
        y: numpy array, shape=(n_samples, output_size)
        batch_size: int, 批次大小
        shuffle: bool, 是否随机打乱
    
    Yields:
        X_batch: shape=(seq_len, input_size, batch_size)
        y_batch: shape=(output_size, batch_size)
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = X[batch_indices]  # (batch_size, seq_len, input_size)
        y_batch = y[batch_indices]  # (batch_size, output_size)
        
        # 转置为模型需要的格式
        X_batch = np.transpose(X_batch, (1, 2, 0))  # (seq_len, input_size, batch_size)
        y_batch = y_batch.T  # (output_size, batch_size)
        
        yield X_batch, y_batch


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载器:\n")
    
    # 加载数据
    data_scaled, scaler, raw_data = load_yahoo_stock(ticker='AAPL', use_mock_data=True)
    
    # 创建序列
    seq_len = 60
    X, y = create_sequences(data_scaled, seq_len=seq_len)
    print(f"\n序列形状: X={X.shape}, y={y.shape}")
    
    # 划分数据
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    # 测试批次生成器
    print(f"\n测试批次生成器 (batch_size=32):")
    batch_count = 0
    for X_batch, y_batch in batch_generator(X_train, y_train, batch_size=32):
        batch_count += 1
        if batch_count == 1:
            print(f"  批次形状: X_batch={X_batch.shape}, y_batch={y_batch.shape}")
    print(f"  总批次数: {batch_count}")
    
    print("\n数据加载器测试成功!")
