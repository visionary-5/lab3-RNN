"""
生成可视化图表的运行脚本
"""
import sys
import os
import glob

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from visualize import generate_all_plots
from utils.data_loader import load_yahoo_stock, create_sequences, split_data


def main():
    """主函数：生成所有可视化图表"""
    
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)
    
    # 1. 查找最新的结果文件
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    if not os.path.exists(results_dir):
        print(f"\n错误: 结果目录不存在: {results_dir}")
        print("请先运行 benchmark.py 生成结果文件")
        return
    
    # 查找所有结果文件
    result_files = glob.glob(os.path.join(results_dir, 'benchmark_results_*.json'))
    
    if not result_files:
        print(f"\n错误: 在 {results_dir} 中未找到结果文件")
        print("请先运行 benchmark.py 生成结果文件")
        return
    
    # 使用最新的结果文件
    latest_result = max(result_files, key=os.path.getmtime)
    print(f"\n找到结果文件: {os.path.basename(latest_result)}")
    
    # 2. 重新加载数据以获取测试集
    print("\n重新加载数据...")
    data_scaled, scaler, raw_data = load_yahoo_stock(ticker='AAPL', use_mock_data=True)
    X, y = create_sequences(data_scaled, seq_len=60)
    _, _, _, _, X_test, y_test = split_data(X, y, train_ratio=0.7, val_ratio=0.15)
    
    print(f"测试集大小: {len(y_test)} 样本")
    
    # 3. 生成所有图表
    print("\n开始生成可视化图表...")
    generate_all_plots(latest_result, y_test, scaler, output_dir=results_dir)
    
    print("\n" + "=" * 80)
    print("✅ 所有图表生成完成！")
    print("=" * 80)
    print(f"\n图表保存在: {results_dir}")
    print("\n生成的图表包括:")
    print("  1. loss_curves_*.png - 收敛曲线对比图")
    print("  2. predictions_*.png - 预测结果对比图")
    print("  3. fusion_weight_*.png - 融合权重分析图")
    print("  4. error_distribution_*.png - 误差分布图")
    print("  5. metrics_comparison_*.png - 指标对比图")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
