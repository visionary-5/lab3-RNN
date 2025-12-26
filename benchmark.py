"""
对比实验框架
同时训练和评估三个模型：Pure Mamba, Pure MinGRU, Hybrid Mamba-GRU
"""
import numpy as np
import sys
import os
import json
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from models import Mamba, MinGRU, MambaGRU
from utils.data_loader import load_yahoo_stock, create_sequences, split_data, batch_generator
from utils.optimizers import AdamOptimizer
from utils.schedulers import CosineAnnealingLR
from utils.regularization import gradient_clipping, l2_regularization_loss, l2_regularization_grad, EarlyStopping
from utils.metrics import compute_all_metrics
from train import train_model, evaluate_model


def run_benchmark(seq_len=60, hidden_size=64, state_size=64,
                  epochs=100, batch_size=32, learning_rate=0.001,
                  use_mock_data=False, save_results=True):
    """
    运行对比实验
    
    参数:
        seq_len: int, 输入序列长度
        hidden_size: int, 隐藏层维度
        state_size: int, Mamba 的状态空间维度
        epochs: int, 训练轮数
        batch_size: int, 批次大小
        learning_rate: float, 学习率
        use_mock_data: bool, 是否使用模拟数据
        save_results: bool, 是否保存结果
    
    返回:
        results: dict, 包含所有模型的结果
    """
    print("\n" + "=" * 100)
    print("MAMBA-GRU 混合架构对比实验")
    print("=" * 100)
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"序列长度: {seq_len}")
    print(f"隐藏层维度: {hidden_size}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print("=" * 100 + "\n")
    
    # ========== 1. 加载和预处理数据 ==========
    print("步骤 1: 加载数据...")
    data_scaled, scaler, raw_data = load_yahoo_stock(ticker='AAPL', use_mock_data=use_mock_data)
    
    print(f"\n步骤 2: 创建序列 (seq_len={seq_len})...")
    X, y = create_sequences(data_scaled, seq_len=seq_len)
    
    print(f"\n步骤 3: 划分数据...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15
    )
    
    # 存储所有模型的结果
    results = {
        'metadata': {
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'state_size': state_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'models': {}
    }
    
    # 定义要对比的模型
    model_configs = [
        {
            'name': 'Pure Mamba',
            'model_class': Mamba,
            'params': {
                'input_size': 1,
                'hidden_size': hidden_size,
                'output_size': 1,
                'state_size': state_size,
                'seed': 42
            }
        },
        {
            'name': 'Pure MinGRU',
            'model_class': MinGRU,
            'params': {
                'input_size': 1,
                'hidden_size': hidden_size,
                'output_size': 1,
                'seed': 42
            }
        },
        {
            'name': 'Hybrid Mamba-GRU',
            'model_class': MambaGRU,
            'params': {
                'input_size': 1,
                'hidden_size': hidden_size,
                'output_size': 1,
                'state_size': state_size,
                'use_vector_alpha': False,
                'seed': 42
            }
        }
    ]
    
    # ========== 2. 训练和评估每个模型 ==========
    for config in model_configs:
        model_name = config['name']
        print("\n" + "=" * 100)
        print(f"训练模型: {model_name}")
        print("=" * 100)
        
        # 初始化模型
        model = config['model_class'](**config['params'])
        
        # 初始化优化器和调度器
        optimizer = AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
        
        # 训练模型
        start_time = time.time()
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            batch_size=batch_size,
            gradient_clip_norm=5.0,
            l2_lambda=0.0001,
            early_stopping_patience=20,
            verbose=True
        )
        training_time = time.time() - start_time
        
        # 评估模型
        predictions, metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            scaler=scaler,
            batch_size=batch_size
        )
        
        # 保存结果
        results['models'][model_name] = {
            'history': history,
            'metrics': metrics,
            'training_time': training_time,
            'predictions': predictions.tolist() if save_results else None
        }
        
        # 如果是混合模型，保存 alpha 历史
        if hasattr(model, 'alpha_history'):
            results['models'][model_name]['alpha_history'] = model.alpha_history
            results['models'][model_name]['final_alpha'] = float(np.mean(model.get_fusion_weight()))
    
    # ========== 3. 打印对比结果 ==========
    print("\n" + "=" * 100)
    print("对比实验结果汇总")
    print("=" * 100)
    
    # 打印 Markdown 表格
    print("\n## 测试集评价指标对比\n")
    print("| 模型 | MSE | RMSE | MAE | R² | MAPE (%) | 训练时间 (s) |")
    print("|------|-----|------|-----|-----|----------|--------------|")
    
    for model_name in ['Pure Mamba', 'Pure MinGRU', 'Hybrid Mamba-GRU']:
        metrics = results['models'][model_name]['metrics']
        train_time = results['models'][model_name]['training_time']
        print(f"| {model_name} | "
              f"{metrics['MSE']:.6f} | "
              f"{metrics['RMSE']:.6f} | "
              f"{metrics['MAE']:.6f} | "
              f"{metrics['R2']:.6f} | "
              f"{metrics['MAPE']:.2f} | "
              f"{train_time:.2f} |")
    
    print("\n" + "=" * 100)
    
    # 找出最佳模型
    best_model = min(results['models'].items(), key=lambda x: x[1]['metrics']['MSE'])
    print(f"\n🏆 最佳模型: {best_model[0]}")
    print(f"   测试集 MSE: {best_model[1]['metrics']['MSE']:.6f}")
    print(f"   测试集 R²: {best_model[1]['metrics']['R2']:.6f}")
    
    # 如果是混合模型，打印融合权重
    if 'Hybrid Mamba-GRU' in results['models']:
        final_alpha = results['models']['Hybrid Mamba-GRU'].get('final_alpha', None)
        if final_alpha is not None:
            print(f"\n📊 混合模型最终融合权重:")
            print(f"   Mamba: {final_alpha:.4f}")
            print(f"   GRU:   {1 - final_alpha:.4f}")
    
    print("\n" + "=" * 100 + "\n")
    
    # ========== 4. 保存结果 ==========
    if save_results:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'benchmark_results_{timestamp}.json')
        
        # 保存为 JSON (需要转换 numpy 类型)
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_to_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {results_file}")
    
    return results


if __name__ == "__main__":
    # 运行对比实验
    results = run_benchmark(
        seq_len=60,
        hidden_size=64,
        state_size=64,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        use_mock_data=True,  # 使用模拟数据以便快速测试
        save_results=True
    )
    
    print("\n对比实验完成！")
