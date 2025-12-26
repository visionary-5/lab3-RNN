# Mamba-GRU 混合架构研究

## 项目简介

本项目实现了一个创新的混合神经网络架构，结合了 **Mamba (选择性状态空间模型)** 和 **MinGRU (最小化门控循环单元)**，用于金融时间序列预测任务。

### 研究主题
**混合架构带来的互补性研究**

### 论文题目
**Mamba-GRU: A Hybrid Architecture Combining Selective State Spaces and Gated Recurrence for Financial Time Series Forecasting**

（Mamba-GRU：一种结合选择性状态空间与门控循环的混合架构在金融时间序列预测中的应用）

## 核心动机 (Motivation)

- **Mamba (SSM)** 擅长捕捉**长期依赖 (Long-range dependency)**，能够看到很远的历史信息
- **RNN (MinGRU)** 通常在捕捉**局部波动 (Short-term dynamics)** 和非线性特征上表现稳健
- 金融数据往往同时包含长期趋势（适合 Mamba）和短期剧烈震荡（适合 GRU）
- 单独使用谁都有短板，因此提出混合架构

## 创新点 (Methodology)

### 并行加权融合架构

```
        Input
          |
        /   \
    Mamba  MinGRU
        \   /
         α融合
          |
        Output
```

- **并行结构**: 输入同时传给 Mamba 和 MinGRU 分支
- **可学习融合权重**: 使用参数 α 控制两个分支的贡献度
  - `output = sigmoid(α) * out_mamba + (1 - sigmoid(α)) * out_gru`
- **自适应学习**: 模型在训练过程中自动学习最优的融合策略

## 项目结构

```
labs-rnn-improve/
│
├── models/                 # 模型实现
│   ├── __init__.py
│   ├── mamba.py           # Mamba 模型
│   ├── min_gru.py         # MinGRU 模型
│   └── mamba_gru.py       # 混合模型 (核心创新)
│
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── data_loader.py     # 数据加载和预处理
│   ├── optimizers.py      # Adam 优化器
│   ├── schedulers.py      # 学习率调度器
│   ├── regularization.py  # 正则化工具
│   └── metrics.py         # 评价指标
│
├── results/               # 实验结果和图表
│
├── main.py                # 主运行脚本
├── train.py               # 训练脚本
├── benchmark.py           # 对比实验框架
├── visualize.py           # 可视化模块
├── requirements.txt       # 依赖包
└── README.md             # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 运行完整实验

```bash
python main.py
```

这将执行以下步骤：
1. 训练三个模型（Pure Mamba, Pure MinGRU, Hybrid Mamba-GRU）
2. 生成对比评价指标
3. 生成高质量可视化图表
4. 生成实验报告

### 2. 仅运行对比实验

```bash
python benchmark.py
```

### 3. 单独训练某个模型

```python
from models import MambaGRU
from utils.data_loader import load_yahoo_stock, create_sequences, split_data
from train import train_model

# 加载数据
data_scaled, scaler, raw_data = load_yahoo_stock('AAPL')
X, y = create_sequences(data_scaled, seq_len=60)
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

# 初始化模型
model = MambaGRU(input_size=1, hidden_size=64, output_size=1)

# 训练
history = train_model(model, X_train, y_train, X_val, y_val, ...)
```

## 实验设计

### 对比实验 (Ablation Study)

对比以下三个模型的性能：
1. **Pure Mamba**: 单独使用 Mamba
2. **Pure MinGRU**: 单独使用 MinGRU
3. **Hybrid Mamba-GRU**: 混合架构

### 评价指标

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **MAPE**: 平均绝对百分比误差

## 可视化输出

实验会生成以下高质量图表：

1. **收敛曲线对比图**: 展示三个模型的训练和验证损失
2. **预测结果对比图**: 在测试集上对比真实值和预测值（含局部放大）
3. **融合权重分析图**: 显示混合模型的 α 权重演化过程
4. **误差分布图**: 展示每个模型的预测误差分布
5. **指标对比柱状图**: 直观对比各项评价指标

## 核心实现细节

### 融合层的梯度推导

前向传播：
```
α_sigmoid = sigmoid(α)
output = α_sigmoid * out_mamba + (1 - α_sigmoid) * out_gru
```

反向传播：
```python
# 对 Mamba 输出的梯度
dout_mamba = dout_fused * α_sigmoid

# 对 GRU 输出的梯度
dout_gru = dout_fused * (1 - α_sigmoid)

# 对 α 的梯度
dalpha_sigmoid = dout_fused * (out_mamba - out_gru)
dalpha = dalpha_sigmoid * α_sigmoid * (1 - α_sigmoid)
```

### 优化技巧

- **Adam 优化器**: 自适应学习率
- **余弦退火学习率**: 平滑衰减
- **梯度裁剪**: 防止梯度爆炸
- **L2 正则化**: 防止过拟合
- **Early Stopping**: 避免过度训练

## 技术特点

✅ **纯 NumPy 实现**: 所有模型都是从零开始实现，深入理解底层原理

✅ **完整的反向传播**: 包含详细的数学推导和梯度计算

✅ **模块化设计**: 代码结构清晰，易于扩展

✅ **学术级可视化**: 生成论文质量的图表

✅ **详细注释**: 关键步骤都有数学公式和解释

## 实验结果示例

| 模型 | MSE | RMSE | MAE | R² | MAPE (%) |
|------|-----|------|-----|-----|----------|
| Pure Mamba | 0.000234 | 0.015301 | 0.011245 | 0.8234 | 2.45 |
| Pure MinGRU | 0.000198 | 0.014071 | 0.010123 | 0.8567 | 2.12 |
| **Hybrid Mamba-GRU** | **0.000156** | **0.012490** | **0.008934** | **0.8923** | **1.87** |

*注: 以上为示例数据，实际结果会因数据和超参数而异*

## 未来工作

- [ ] 在更多金融数据集上进行验证
- [ ] 探索 Attention 融合机制
- [ ] 分析不同序列长度的影响
- [ ] 多步预测扩展
- [ ] 在线学习支持

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{mamba-gru-2024,
  title={Mamba-GRU: A Hybrid Architecture for Financial Time Series Forecasting},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/mamba-gru}}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**关键词**: Mamba, MinGRU, 混合架构, 时间序列预测, 金融数据, 状态空间模型, 门控循环单元, 深度学习, NumPy
