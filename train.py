"""
训练脚本
实现完整的训练流程，包括 Adam 优化器、学习率调度、Early Stopping、梯度裁剪等
"""
import numpy as np
import sys
import os
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_yahoo_stock, create_sequences, split_data, batch_generator
from utils.optimizers import AdamOptimizer
from utils.schedulers import CosineAnnealingLR
from utils.regularization import gradient_clipping, l2_regularization_loss, l2_regularization_grad, EarlyStopping
from utils.metrics import compute_all_metrics


def train_model(model, X_train, y_train, X_val, y_val, 
                optimizer, scheduler=None, 
                epochs=50, batch_size=32,
                gradient_clip_norm=5.0, l2_lambda=0.0001,
                early_stopping_patience=15, verbose=True):
    """
    训练模型
    
    参数:
        model: 模型对象 (Mamba, MinGRU 或 MambaGRU)
        X_train: 训练集输入
        y_train: 训练集标签
        X_val: 验证集输入
        y_val: 验证集标签
        optimizer: 优化器对象
        scheduler: 学习率调度器 (可选)
        epochs: 训练轮数
        batch_size: 批次大小
        gradient_clip_norm: 梯度裁剪的最大范数
        l2_lambda: L2 正则化系数
        early_stopping_patience: 早停耐心值
        verbose: 是否打印详细信息
    
    返回:
        history: dict, 包含训练历史
    """
    # 初始化历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'gradient_norm': [],
        'epoch_time': []
    }
    
    # 初始化 Early Stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-6,
        verbose=verbose
    )
    
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"批次大小: {batch_size}")
    print(f"总轮数: {epochs}")
    print(f"优化器: {optimizer.__class__.__name__}")
    print(f"学习率调度器: {scheduler.__class__.__name__ if scheduler else 'None'}")
    print(f"梯度裁剪范数: {gradient_clip_norm}")
    print(f"L2 正则化系数: {l2_lambda}")
    print("=" * 80 + "\n")
    
    best_val_loss = float('inf')
    best_model_params = None
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # ========== 训练阶段 ==========
        train_losses = []
        batch_count = 0
        
        for X_batch, y_batch in batch_generator(X_train, y_train, batch_size=batch_size, shuffle=True):
            # 前向传播
            y_pred, cache = model.forward(X_batch)
            
            # 计算损失 (MSE + L2 正则化)
            mse_loss, dy = model.compute_loss(y_pred, y_batch)
            
            # L2 正则化损失
            params = model.get_parameters()
            l2_loss = l2_regularization_loss(params, lambda_reg=l2_lambda)
            total_loss = mse_loss + l2_loss
            
            train_losses.append(mse_loss)  # 记录 MSE 损失
            
            # 反向传播
            dx = model.backward(dy, cache)
            
            # 获取梯度
            grads = model.get_gradients()
            
            # 添加 L2 正则化的梯度
            if l2_lambda > 0:
                l2_grads = l2_regularization_grad(params, lambda_reg=l2_lambda)
                for key in grads:
                    if key in l2_grads:
                        grads[key] = grads[key] + l2_grads[key]
            
            # 梯度裁剪
            grads, grad_norm = gradient_clipping(grads, max_norm=gradient_clip_norm)
            
            # 更新参数
            updated_params = optimizer.update(params, grads)
            model.set_parameters(updated_params)
            
            batch_count += 1
        
        avg_train_loss = np.mean(train_losses)
        
        # ========== 验证阶段 ==========
        val_losses = []
        for X_batch, y_batch in batch_generator(X_val, y_val, batch_size=batch_size, shuffle=False):
            y_pred, _ = model.forward(X_batch)
            loss, _ = model.compute_loss(y_pred, y_batch)
            val_losses.append(loss)
        
        avg_val_loss = np.mean(val_losses)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(optimizer.learning_rate)
        history['gradient_norm'].append(grad_norm if batch_count > 0 else 0)
        epoch_time = time.time() - epoch_start_time
        history['epoch_time'].append(epoch_time)
        
        # 打印进度
        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {optimizer.learning_rate:.6f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_params = {k: v.copy() for k, v in model.get_parameters().items()}
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        # Early Stopping 检查
        if early_stopping(avg_val_loss, epoch):
            print(f"\n早停触发！在 Epoch {epoch + 1} 停止训练")
            break
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("=" * 80 + "\n")
    
    # 恢复最佳模型参数
    if best_model_params is not None:
        model.set_parameters(best_model_params)
    
    return history


def evaluate_model(model, X_test, y_test, scaler=None, batch_size=32):
    """
    评估模型
    
    参数:
        model: 模型对象
        X_test: 测试集输入
        y_test: 测试集标签
        scaler: MinMaxScaler 对象，用于反归一化
        batch_size: 批次大小
    
    返回:
        predictions: numpy array, 预测值
        metrics: dict, 评价指标
    """
    print("\n" + "=" * 80)
    print("评估模型")
    print("=" * 80)
    
    # 收集所有预测结果
    all_predictions = []
    all_targets = []
    
    for X_batch, y_batch in batch_generator(X_test, y_test, batch_size=batch_size, shuffle=False):
        y_pred, _ = model.forward(X_batch)
        all_predictions.append(y_pred.flatten())
        all_targets.append(y_batch.flatten())
    
    # 合并所有批次的结果
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # 如果提供了 scaler，则反归一化
    if scaler is not None:
        # 重塑为 (n_samples, 1) 以便使用 scaler
        predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        targets_original = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        # 计算原始尺度的指标
        metrics = compute_all_metrics(targets_original, predictions_original)
        print(f"评价指标 (原始尺度):")
    else:
        metrics = compute_all_metrics(targets, predictions)
        print(f"评价指标 (归一化尺度):")
    
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  R²:   {metrics['R2']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print("=" * 80 + "\n")
    
    return predictions, metrics


if __name__ == "__main__":
    from models import Mamba, MinGRU, MambaGRU
    
    print("\n" + "=" * 80)
    print("训练脚本测试")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n步骤 1: 加载数据...")
    data_scaled, scaler, raw_data = load_yahoo_stock(ticker='AAPL', use_mock_data=True)
    
    # 2. 创建序列
    print("\n步骤 2: 创建序列...")
    seq_len = 30
    X, y = create_sequences(data_scaled, seq_len=seq_len)
    
    # 3. 划分数据
    print("\n步骤 3: 划分数据...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, train_ratio=0.7, val_ratio=0.15)
    
    # 4. 初始化模型
    print("\n步骤 4: 初始化模型...")
    model = MambaGRU(
        input_size=1,
        hidden_size=32,
        output_size=1,
        state_size=32,
        use_vector_alpha=False,
        seed=42
    )
    
    # 5. 初始化优化器和调度器
    print("\n步骤 5: 初始化优化器...")
    optimizer = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)
    
    # 6. 训练模型
    print("\n步骤 6: 训练模型...")
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=10,  # 测试时使用较少的 epoch
        batch_size=32,
        gradient_clip_norm=5.0,
        l2_lambda=0.0001,
        early_stopping_patience=5,
        verbose=True
    )
    
    # 7. 评估模型
    print("\n步骤 7: 评估模型...")
    predictions, metrics = evaluate_model(model, X_test, y_test, scaler=scaler, batch_size=32)
    
    print("\n训练脚本测试完成!")
