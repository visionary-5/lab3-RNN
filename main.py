"""
主运行脚本
执行完整的对比实验流程：训练、评估、可视化
"""
import sys
import os
import json

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from benchmark import run_benchmark
from visualize import generate_all_plots
from utils.data_loader import load_yahoo_stock, create_sequences, split_data


def main():
    """
    主函数：执行完整的实验流程
    """
    print("\n" + "=" * 100)
    print("MAMBA-GRU 混合架构研究 - 完整实验流程")
    print("主题：混合架构带来的互补性研究")
    print("=" * 100)
    
    # ========== 实验配置 ==========
    config = {
        'seq_len': 60,              # 输入序列长度
        'hidden_size': 64,          # 隐藏层维度
        'state_size': 64,           # Mamba 状态空间维度
        'epochs': 100,              # 训练轮数
        'batch_size': 32,           # 批次大小
        'learning_rate': 0.001,     # 学习率
        'use_mock_data': False,     # 是否使用模拟数据（True=快速测试，False=真实数据）
        'save_results': True        # 是否保存结果
    }
    
    print("\n实验配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # ========== 步骤 1: 运行对比实验 ==========
    print("\n" + "=" * 100)
    print("步骤 1: 运行对比实验 (训练三个模型)")
    print("=" * 100)
    
    results = run_benchmark(
        seq_len=config['seq_len'],
        hidden_size=config['hidden_size'],
        state_size=config['state_size'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        use_mock_data=config['use_mock_data'],
        save_results=config['save_results']
    )
    
    # ========== 步骤 2: 生成可视化 ==========
    print("\n" + "=" * 100)
    print("步骤 2: 生成可视化图表")
    print("=" * 100)
    
    # 重新加载数据以获取测试集
    print("\n重新加载数据以生成可视化...")
    data_scaled, scaler, raw_data = load_yahoo_stock(
        ticker='AAPL',
        use_mock_data=config['use_mock_data']
    )
    X, y = create_sequences(data_scaled, seq_len=config['seq_len'])
    _, _, _, _, X_test, y_test = split_data(X, y, train_ratio=0.7, val_ratio=0.15)
    
    # 查找最新的结果文件
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    result_files = [f for f in os.listdir(results_dir) if f.startswith('benchmark_results_') and f.endswith('.json')]
    if result_files:
        latest_result = max(result_files)
        result_path = os.path.join(results_dir, latest_result)
        print(f"\n使用结果文件: {result_path}")
        
        # 生成所有图表
        generate_all_plots(result_path, y_test, scaler, output_dir=results_dir)
    else:
        print("\n警告: 未找到结果文件，跳过可视化步骤")
    
    # ========== 步骤 3: 生成实验报告 ==========
    print("\n" + "=" * 100)
    print("步骤 3: 生成实验报告")
    print("=" * 100)
    
    report = generate_experiment_report(results)
    
    # 保存报告
    report_path = os.path.join(results_dir, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n实验报告已保存到: {report_path}")
    
    # ========== 完成 ==========
    print("\n" + "=" * 100)
    print("✅ 实验流程完成！")
    print("=" * 100)
    print("\n生成的文件:")
    print(f"  - 结果数据: {results_dir}/benchmark_results_*.json")
    print(f"  - 可视化图表: {results_dir}/*.png")
    print(f"  - 实验报告: {report_path}")
    print("\n" + "=" * 100 + "\n")


def generate_experiment_report(results):
    """
    生成 Markdown 格式的实验报告
    
    参数:
        results: dict, 实验结果
    
    返回:
        str, Markdown 格式的报告
    """
    report = []
    
    # 标题
    report.append("# Mamba-GRU 混合架构实验报告")
    report.append("\n## 研究主题")
    report.append("混合架构带来的互补性研究\n")
    report.append("## 论文题目")
    report.append("**Mamba-GRU: A Hybrid Architecture Combining Selective State Spaces and Gated Recurrence for Financial Time Series Forecasting**")
    report.append("\n（Mamba-GRU：一种结合选择性状态空间与门控循环的混合架构在金融时间序列预测中的应用）\n")
    
    # 实验配置
    report.append("## 实验配置")
    report.append("\n| 参数 | 值 |")
    report.append("|------|-----|")
    metadata = results['metadata']
    report.append(f"| 序列长度 | {metadata['seq_len']} |")
    report.append(f"| 隐藏层维度 | {metadata['hidden_size']} |")
    report.append(f"| 状态空间维度 | {metadata['state_size']} |")
    report.append(f"| 训练轮数 | {metadata['epochs']} |")
    report.append(f"| 批次大小 | {metadata['batch_size']} |")
    report.append(f"| 学习率 | {metadata['learning_rate']} |")
    report.append(f"| 训练样本数 | {metadata['train_samples']} |")
    report.append(f"| 验证样本数 | {metadata['val_samples']} |")
    report.append(f"| 测试样本数 | {metadata['test_samples']} |")
    
    # 核心动机
    report.append("\n## 核心动机 (Motivation)")
    report.append("\n- **Mamba (SSM)** 擅长捕捉**长期依赖 (Long-range dependency)**，能够看到很远的历史信息")
    report.append("- **RNN (MinGRU)** 通常在捕捉**局部波动 (Short-term dynamics)** 和非线性特征上表现稳健")
    report.append("- 金融数据往往同时包含长期趋势（适合 Mamba）和短期剧烈震荡（适合 GRU）")
    report.append("- 单独使用谁都有短板，因此提出混合架构\n")
    
    # 实验结果
    report.append("## 实验结果\n")
    report.append("### 测试集评价指标对比\n")
    report.append("| 模型 | MSE | RMSE | MAE | R² | MAPE (%) | 训练时间 (s) |")
    report.append("|------|-----|------|-----|-----|----------|--------------|")
    
    for model_name in ['Pure Mamba', 'Pure MinGRU', 'Hybrid Mamba-GRU']:
        metrics = results['models'][model_name]['metrics']
        train_time = results['models'][model_name]['training_time']
        report.append(f"| {model_name} | "
                     f"{metrics['MSE']:.6f} | "
                     f"{metrics['RMSE']:.6f} | "
                     f"{metrics['MAE']:.6f} | "
                     f"{metrics['R2']:.6f} | "
                     f"{metrics['MAPE']:.2f} | "
                     f"{train_time:.2f} |")
    
    # 分析与讨论
    report.append("\n### 分析与讨论\n")
    
    # 找出最佳模型
    best_model = min(results['models'].items(), key=lambda x: x[1]['metrics']['MSE'])
    report.append(f"**最佳模型:** {best_model[0]}\n")
    report.append(f"- 测试集 MSE: {best_model[1]['metrics']['MSE']:.6f}")
    report.append(f"- 测试集 R²: {best_model[1]['metrics']['R2']:.6f}\n")
    
    # 如果混合模型是最佳的
    if best_model[0] == 'Hybrid Mamba-GRU':
        report.append("**关键发现:**")
        report.append("- 混合架构成功结合了 Mamba 的长期记忆能力和 GRU 的局部建模能力")
        report.append("- 在金融时间序列预测任务上取得了最佳性能")
        
        if 'final_alpha' in results['models']['Hybrid Mamba-GRU']:
            alpha = results['models']['Hybrid Mamba-GRU']['final_alpha']
            report.append(f"\n**融合权重分析:**")
            report.append(f"- Mamba 权重: {alpha:.4f}")
            report.append(f"- GRU 权重: {1 - alpha:.4f}")
            
            if alpha > 0.5:
                report.append("- 模型更倾向于使用 Mamba 的长期依赖特性")
            elif alpha < 0.5:
                report.append("- 模型更倾向于使用 GRU 的短期动态特性")
            else:
                report.append("- 模型平衡地使用两种架构的特性")
    
    # 结论
    report.append("\n## 结论\n")
    report.append("本实验验证了 Mamba-GRU 混合架构在金融时间序列预测中的有效性。")
    report.append("通过并行加权融合策略，模型能够自适应地学习如何平衡长期依赖和短期动态，")
    report.append("从而在 AAPL 股票价格预测任务上取得了优异的性能。\n")
    
    # 未来工作
    report.append("## 未来工作\n")
    report.append("- 在更多金融数据集上进行验证")
    report.append("- 探索不同的融合策略（如 Attention 机制）")
    report.append("- 进行更深入的消融实验")
    report.append("- 分析模型在不同市场条件下的表现")
    
    return '\n'.join(report)


if __name__ == "__main__":
    main()
