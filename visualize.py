"""
高质量可视化模块
生成学术论文级别的图表，包括：
1. 收敛曲线对比图
2. 预测结果对比图
3. 融合权重分析图
4. 误差分布图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
from datetime import datetime

# 设置 Matplotlib 中文支持和学术风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# 学术论文配色方案
COLORS = {
    'Pure Mamba': '#1f77b4',      # 蓝色
    'Pure MinGRU': '#ff7f0e',     # 橙色
    'Hybrid Mamba-GRU': '#2ca02c'  # 绿色
}


def plot_loss_curves(results, save_path=None):
    """
    绘制收敛曲线对比图
    
    参数:
        results: dict, 对比实验的结果
        save_path: str, 保存路径 (可选)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制训练损失
    for model_name, model_data in results['models'].items():
        history = model_data['history']
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4, markevery=max(1, len(epochs)//20))
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss (MSE)', fontweight='bold')
    ax1.set_title('Training Loss Convergence', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_yscale('log')  # 对数坐标更好地展示收敛过程
    
    # 绘制验证损失
    for model_name, model_data in results['models'].items():
        history = model_data['history']
        epochs = range(1, len(history['val_loss']) + 1)
        ax2.plot(epochs, history['val_loss'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='s', markersize=4, markevery=max(1, len(epochs)//20))
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Validation Loss (MSE)', fontweight='bold')
    ax2.set_title('Validation Loss Convergence', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"收敛曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predictions_comparison(results, y_test, scaler, 
                                start_idx=0, end_idx=200, save_path=None):
    """
    绘制预测结果对比图（带局部放大）
    
    参数:
        results: dict, 对比实验的结果
        y_test: numpy array, 测试集真实值
        scaler: MinMaxScaler, 用于反归一化
        start_idx: int, 绘图起始索引
        end_idx: int, 绘图结束索引
        save_path: str, 保存路径 (可选)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # 准备真实值
    y_true_flat = y_test.flatten()
    y_true_original = scaler.inverse_transform(y_true_flat.reshape(-1, 1)).flatten()
    
    # 截取绘图范围
    plot_range = slice(start_idx, min(end_idx, len(y_true_original)))
    x_axis = np.arange(start_idx, start_idx + len(y_true_original[plot_range]))
    
    # 上半部分：完整对比
    ax1.plot(x_axis, y_true_original[plot_range], 
            label='True Values', color='black', linewidth=2, linestyle='--', alpha=0.7)
    
    for model_name, model_data in results['models'].items():
        predictions = np.array(model_data['predictions'])
        pred_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        ax1.plot(x_axis, pred_original[plot_range], 
                label=f'{model_name} Predictions', 
                color=COLORS[model_name], linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('Time Step', fontweight='bold')
    ax1.set_ylabel('Stock Price ($)', fontweight='bold')
    ax1.set_title('Prediction Comparison on Test Set', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 下半部分：局部放大（选择中间一段）
    zoom_start = len(y_true_original[plot_range]) // 3
    zoom_end = zoom_start + 50
    zoom_range = slice(start_idx + zoom_start, start_idx + zoom_end)
    x_zoom = np.arange(start_idx + zoom_start, start_idx + zoom_end)
    
    ax2.plot(x_zoom, y_true_original[zoom_range], 
            label='True Values', color='black', linewidth=3, linestyle='--', marker='o', markersize=5, alpha=0.7)
    
    for model_name, model_data in results['models'].items():
        predictions = np.array(model_data['predictions'])
        pred_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        ax2.plot(x_zoom, pred_original[zoom_range], 
                label=f'{model_name} Predictions', 
                color=COLORS[model_name], linewidth=2, marker='s', markersize=4, alpha=0.8)
    
    ax2.set_xlabel('Time Step', fontweight='bold')
    ax2.set_ylabel('Stock Price ($)', fontweight='bold')
    ax2.set_title('Zoomed-in View (Detail Comparison)', fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, shadow=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"预测对比图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_fusion_weight_evolution(results, save_path=None):
    """
    绘制融合权重 alpha 的演化曲线
    
    参数:
        results: dict, 对比实验的结果
        save_path: str, 保存路径 (可选)
    """
    if 'Hybrid Mamba-GRU' not in results['models']:
        print("警告: 结果中没有混合模型，无法绘制融合权重图")
        return
    
    hybrid_data = results['models']['Hybrid Mamba-GRU']
    if 'alpha_history' not in hybrid_data:
        print("警告: 混合模型没有记录 alpha 历史")
        return
    
    alpha_history = hybrid_data['alpha_history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：alpha 随训练步骤的变化
    steps = np.arange(len(alpha_history))
    ax1.plot(steps, alpha_history, color='#d62728', linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Equal Weight (0.5)')
    ax1.fill_between(steps, alpha_history, 0.5, 
                     where=(np.array(alpha_history) >= 0.5), 
                     color='#1f77b4', alpha=0.3, label='Mamba-dominated')
    ax1.fill_between(steps, alpha_history, 0.5, 
                     where=(np.array(alpha_history) < 0.5), 
                     color='#ff7f0e', alpha=0.3, label='GRU-dominated')
    
    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('Fusion Weight α (Mamba)', fontweight='bold')
    ax1.set_title('Evolution of Fusion Weight During Training', fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim([0, 1])
    
    # 右图：Mamba 和 GRU 的权重对比
    alpha_mean = np.mean(alpha_history)
    gru_weight = 1 - alpha_mean
    
    weights = [alpha_mean, gru_weight]
    labels = ['Mamba Weight', 'GRU Weight']
    colors = ['#1f77b4', '#ff7f0e']
    
    wedges, texts, autotexts = ax2.pie(weights, labels=labels, colors=colors, 
                                        autopct='%1.2f%%', startangle=90,
                                        textprops={'fontsize': 14, 'fontweight': 'bold'},
                                        explode=(0.05, 0.05), shadow=True)
    
    ax2.set_title('Average Fusion Weight Distribution', fontweight='bold')
    
    # 美化百分比文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"融合权重图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(results, y_test, scaler, save_path=None):
    """
    绘制预测误差分布图
    
    参数:
        results: dict, 对比实验的结果
        y_test: numpy array, 测试集真实值
        scaler: MinMaxScaler, 用于反归一化
        save_path: str, 保存路径 (可选)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 准备真实值
    y_true_flat = y_test.flatten()
    y_true_original = scaler.inverse_transform(y_true_flat.reshape(-1, 1)).flatten()
    
    for idx, (model_name, model_data) in enumerate(results['models'].items()):
        predictions = np.array(model_data['predictions'])
        pred_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # 计算误差
        errors = pred_original - y_true_original
        
        # 绘制直方图
        ax = axes[idx]
        n, bins, patches = ax.hist(errors, bins=50, color=COLORS[model_name], 
                                   alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # 添加统计信息
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_error:.2f}')
        ax.axvline(mean_error + std_error, color='orange', linestyle=':', linewidth=2, 
                  label=f'±1σ: {std_error:.2f}')
        ax.axvline(mean_error - std_error, color='orange', linestyle=':', linewidth=2)
        
        ax.set_xlabel('Prediction Error ($)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{model_name}\nError Distribution', fontweight='bold')
        ax.legend(loc='upper right', frameon=True, shadow=True)
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"误差分布图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(results, save_path=None):
    """
    绘制评价指标对比柱状图
    
    参数:
        results: dict, 对比实验的结果
        save_path: str, 保存路径 (可选)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    model_names = list(results['models'].keys())
    metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'R2']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [results['models'][name]['metrics'][metric] for name in model_names]
        colors_list = [COLORS[name] for name in model_names]
        
        bars = ax.bar(model_names, values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # R² 的理想值是 1，其他指标越小越好
        if metric == 'R2':
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Score')
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"指标对比图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_plots(results_file, y_test, scaler, output_dir='results'):
    """
    从结果文件生成所有可视化图表
    
    参数:
        results_file: str, 结果 JSON 文件路径
        y_test: numpy array, 测试集真实值
        scaler: MinMaxScaler, 用于反归一化
        output_dir: str, 输出目录
    """
    # 加载结果
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 收敛曲线
    print("\n1. 生成收敛曲线...")
    plot_loss_curves(results, save_path=os.path.join(output_dir, f'loss_curves_{timestamp}.png'))
    
    # 2. 预测对比
    print("2. 生成预测对比图...")
    plot_predictions_comparison(results, y_test, scaler, 
                               save_path=os.path.join(output_dir, f'predictions_{timestamp}.png'))
    
    # 3. 融合权重
    print("3. 生成融合权重图...")
    plot_fusion_weight_evolution(results, 
                                 save_path=os.path.join(output_dir, f'fusion_weight_{timestamp}.png'))
    
    # 4. 误差分布
    print("4. 生成误差分布图...")
    plot_error_distribution(results, y_test, scaler, 
                           save_path=os.path.join(output_dir, f'error_distribution_{timestamp}.png'))
    
    # 5. 指标对比
    print("5. 生成指标对比图...")
    plot_metrics_comparison(results, 
                           save_path=os.path.join(output_dir, f'metrics_comparison_{timestamp}.png'))
    
    print("\n" + "=" * 80)
    print("所有图表生成完成！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("可视化模块独立测试...")
    print("提示: 请先运行 benchmark.py 生成结果文件，然后使用 generate_all_plots 函数生成图表")
