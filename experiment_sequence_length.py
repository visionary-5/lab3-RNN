"""
序列长度敏感性实验
测试不同序列长度对模型性能的影响，验证混合模型在长序列上的优势
"""
import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(__file__))

from benchmark import run_benchmark


def run_sequence_length_experiment():
    """
    运行序列长度敏感性实验
    """
    print("\n" + "=" * 100)
    print("序列长度敏感性实验")
    print("=" * 100)
    print("\n目标：验证混合模型在不同序列长度上的表现")
    print("假设：随着序列长度增加，混合模型的优势会逐渐显现\n")
    
    # 实验配置
    sequence_lengths = [30, 60, 90, 120]  # 测试不同的序列长度
    results_by_length = {}
    
    for seq_len in sequence_lengths:
        print(f"\n{'='*100}")
        print(f"实验：序列长度 = {seq_len}")
        print(f"{'='*100}\n")
        
        try:
            results = run_benchmark(
                seq_len=seq_len,
                hidden_size=64,
                state_size=64,
                epochs=50,  # 减少 epoch 以加快实验
                batch_size=32,
                learning_rate=0.001,
                use_mock_data=True,  # 使用模拟数据
                save_results=False   # 不保存中间结果
            )
            
            # 提取关键指标
            results_by_length[seq_len] = {
                'Pure Mamba': results['models']['Pure Mamba']['metrics'],
                'Pure MinGRU': results['models']['Pure MinGRU']['metrics'],
                'Hybrid Mamba-GRU': results['models']['Hybrid Mamba-GRU']['metrics']
            }
            
            # 打印当前结果
            print(f"\n✅ 序列长度 {seq_len} 实验完成")
            print(f"   Pure Mamba MSE:   {results['models']['Pure Mamba']['metrics']['MSE']:.4f}")
            print(f"   Pure MinGRU MSE:  {results['models']['Pure MinGRU']['metrics']['MSE']:.4f}")
            print(f"   Hybrid MSE:       {results['models']['Hybrid Mamba-GRU']['metrics']['MSE']:.4f}")
            
        except Exception as e:
            print(f"\n❌ 序列长度 {seq_len} 实验失败: {str(e)}")
            continue
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'sequence_length_experiment.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_by_length, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")
    
    # 可视化结果
    plot_sequence_length_analysis(results_by_length, results_dir)
    
    return results_by_length


def plot_sequence_length_analysis(results_by_length, output_dir):
    """
    绘制序列长度分析图
    """
    print("\n生成序列长度分析图...")
    
    # 准备数据
    seq_lengths = sorted(results_by_length.keys())
    models = ['Pure Mamba', 'Pure MinGRU', 'Hybrid Mamba-GRU']
    colors = {'Pure Mamba': '#1f77b4', 'Pure MinGRU': '#ff7f0e', 'Hybrid Mamba-GRU': '#2ca02c'}
    
    # 提取指标
    metrics_to_plot = ['MSE', 'R2']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for model in models:
            values = [results_by_length[sl][model][metric] for sl in seq_lengths]
            ax.plot(seq_lengths, values, marker='o', linewidth=2.5, 
                   markersize=8, label=model, color=colors[model])
        
        ax.set_xlabel('Sequence Length', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric} vs Sequence Length', fontsize=16, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(seq_lengths)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'sequence_length_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"分析图已保存到: {output_file}")
    plt.close()
    
    # 打印分析结论
    print("\n" + "=" * 100)
    print("实验结论")
    print("=" * 100)
    
    # 计算混合模型相对于最佳单模型的改进
    for seq_len in seq_lengths:
        mamba_mse = results_by_length[seq_len]['Pure Mamba']['MSE']
        gru_mse = results_by_length[seq_len]['Pure MinGRU']['MSE']
        hybrid_mse = results_by_length[seq_len]['Hybrid Mamba-GRU']['MSE']
        
        best_single_mse = min(mamba_mse, gru_mse)
        improvement = (best_single_mse - hybrid_mse) / best_single_mse * 100
        
        print(f"\n序列长度 {seq_len}:")
        print(f"  最佳单模型 MSE: {best_single_mse:.4f}")
        print(f"  混合模型 MSE:   {hybrid_mse:.4f}")
        
        if improvement > 0:
            print(f"  ✅ 混合模型改进: {improvement:.2f}%")
        else:
            print(f"  ❌ 混合模型劣化: {abs(improvement):.2f}%")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    print("\n⚠️  注意：此实验需要较长时间（约20-30分钟）")
    print("建议：先运行一个快速测试，确认代码正常后再运行完整实验\n")
    
    response = input("是否继续？(y/n): ")
    if response.lower() == 'y':
        results = run_sequence_length_experiment()
        print("\n✅ 实验完成！")
    else:
        print("\n已取消实验")
