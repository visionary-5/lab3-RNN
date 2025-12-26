"""
Mamba-GRU 混合模型实现
使用纯 NumPy 实现并行加权融合架构

数学推导：
===========

1. 并行分支结构:
   输入 X 同时传入两个分支：
   - Mamba 分支: out_mamba = Mamba(X)
   - GRU 分支: out_gru = MinGRU(X)

2. 加权融合层 (Weighted Fusion):
   使用可学习的权重参数 alpha (scalar 或 vector) 进行融合：
   
   α_sigmoid = sigmoid(alpha)
   output = α_sigmoid * out_mamba + (1 - α_sigmoid) * out_gru
   
   其中 sigmoid 确保权重在 [0, 1] 范围内

3. 融合层的梯度计算 (关键部分):
   
   前向传播:
   -------
   α_sigmoid = 1 / (1 + exp(-alpha))
   out = α_sigmoid * out_mamba + (1 - α_sigmoid) * out_gru
   
   反向传播:
   -------
   设上游梯度为 dL/dout = dy
   
   (a) 对 alpha 的梯度:
       dL/dα_sigmoid = dy * (out_mamba - out_gru)
       
       由链式法则，sigmoid 的导数为:
       dα_sigmoid/dalpha = α_sigmoid * (1 - α_sigmoid)
       
       因此:
       dL/dalpha = dL/dα_sigmoid * dα_sigmoid/dalpha
                 = dy * (out_mamba - out_gru) * α_sigmoid * (1 - α_sigmoid)
   
   (b) 对 Mamba 输出的梯度:
       dL/dout_mamba = dy * α_sigmoid
       
   (c) 对 GRU 输出的梯度:
       dL/dout_gru = dy * (1 - α_sigmoid)
   
   然后将这些梯度分别传回 Mamba 和 GRU 分支进行反向传播

4. 输出层:
   最终通过线性层映射到预测值:
   y_pred = W_out @ out + b_out
"""
import numpy as np
from .mamba import Mamba
from .min_gru import MinGRU


class MambaGRU:
    """
    Mamba-GRU 混合模型
    
    采用并行加权融合策略：
    - 输入同时传给 Mamba 和 MinGRU 分支
    - 使用可学习的权重 alpha 进行加权融合
    - 通过输出层映射到最终预测
    
    架构:
        Input
          |
        /   \\
    Mamba  MinGRU
        \\   /
         α融合
          |
        Output
    """
    
    def __init__(self, input_size, hidden_size, output_size, 
                 state_size=64, use_vector_alpha=False, seed=42):
        """
        初始化 Mamba-GRU 混合模型
        
        参数:
            input_size: int, 输入特征维度
            hidden_size: int, 隐藏层维度
            output_size: int, 输出维度
            state_size: int, Mamba 的状态空间维度
            use_vector_alpha: bool, 是否使用向量 alpha (True) 或标量 alpha (False)
            seed: int, 随机种子
        """
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_vector_alpha = use_vector_alpha
        
        # 初始化 Mamba 分支
        self.mamba = Mamba(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,  # 输出到融合层
            state_size=state_size,
            seed=seed
        )
        
        # 初始化 MinGRU 分支
        self.min_gru = MinGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,  # 输出到融合层
            seed=seed + 1
        )
        
        # 初始化融合权重 alpha
        # alpha 的初始值设为 0，使得初始时 sigmoid(alpha) = 0.5 (均等融合)
        if use_vector_alpha:
            # 向量 alpha: 每个隐藏维度有独立的权重
            self.alpha = np.zeros((hidden_size, 1))
        else:
            # 标量 alpha: 所有维度共享同一个权重
            self.alpha = np.array([[0.0]])
        
        # 最终输出层
        self.W_out = np.random.randn(output_size, hidden_size) * 0.01
        self.b_out = np.zeros((output_size, 1))
        
        # 重置梯度
        self.reset_gradients()
        
        # 记录融合权重的历史 (用于可视化)
        self.alpha_history = []
    
    def reset_gradients(self):
        """重置所有梯度为零"""
        self.dalpha = np.zeros_like(self.alpha)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)
        
        # 子模块的梯度由它们自己管理
        self.mamba.reset_gradients()
        self.min_gru.reset_gradients()
    
    def sigmoid(self, x):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: numpy array, shape=(seq_len, input_size, batch_size)
        
        返回:
            y_pred: numpy array, shape=(output_size, batch_size), 最终输出
            cache: dict, 包含前向传播过程中的所有中间变量
        """
        # 1. 并行通过两个分支
        out_mamba, cache_mamba = self.mamba.forward(X)  # shape: (hidden_size, batch_size)
        out_gru, cache_gru = self.min_gru.forward(X)    # shape: (hidden_size, batch_size)
        
        # 2. 计算融合权重
        alpha_sigmoid = self.sigmoid(self.alpha)  # shape: (hidden_size, 1) 或 (1, 1)
        
        # 3. 加权融合
        # out_fused = α * out_mamba + (1 - α) * out_gru
        out_fused = alpha_sigmoid * out_mamba + (1 - alpha_sigmoid) * out_gru
        
        # 4. 输出层
        y_pred = self.W_out @ out_fused + self.b_out
        
        # 保存中间变量用于反向传播
        cache = {
            'X': X,
            'out_mamba': out_mamba,
            'out_gru': out_gru,
            'alpha_sigmoid': alpha_sigmoid,
            'out_fused': out_fused,
            'cache_mamba': cache_mamba,
            'cache_gru': cache_gru
        }
        
        # 记录 alpha 的值（用于可视化）
        self.alpha_history.append(np.mean(alpha_sigmoid))
        
        return y_pred, cache
    
    def backward(self, dy, cache):
        """
        反向传播
        
        参数:
            dy: numpy array, shape=(output_size, batch_size), 输出层的梯度
            cache: dict, 前向传播时保存的中间变量
        
        返回:
            dx_combined: numpy array, shape=(seq_len, input_size, batch_size), 输入的梯度
        """
        # 提取缓存的中间变量
        out_mamba = cache['out_mamba']
        out_gru = cache['out_gru']
        alpha_sigmoid = cache['alpha_sigmoid']
        out_fused = cache['out_fused']
        
        # 重置梯度
        self.reset_gradients()
        
        # 1. 输出层的梯度
        # y_pred = W_out @ out_fused + b_out
        self.dW_out = dy @ out_fused.T
        self.db_out = np.sum(dy, axis=1, keepdims=True)
        
        # 传递到融合层的梯度
        dout_fused = self.W_out.T @ dy  # shape: (hidden_size, batch_size)
        
        # 2. 融合层的梯度
        # out_fused = α_sigmoid * out_mamba + (1 - α_sigmoid) * out_gru
        
        # (a) 对 Mamba 输出的梯度
        dout_mamba = dout_fused * alpha_sigmoid  # shape: (hidden_size, batch_size)
        
        # (b) 对 GRU 输出的梯度
        dout_gru = dout_fused * (1 - alpha_sigmoid)  # shape: (hidden_size, batch_size)
        
        # (c) 对 alpha 的梯度
        # dL/dα_sigmoid = dout_fused * (out_mamba - out_gru)
        dalpha_sigmoid = dout_fused * (out_mamba - out_gru)  # shape: (hidden_size, batch_size)
        
        # 对标量/向量 alpha 求和
        dalpha_sigmoid = np.sum(dalpha_sigmoid, axis=1, keepdims=True)  # shape: (hidden_size, 1) 或 (1, 1)
        if not self.use_vector_alpha:
            dalpha_sigmoid = np.sum(dalpha_sigmoid, keepdims=True)  # 标量情况下全部求和
        
        # 通过 sigmoid 的导数
        # dα_sigmoid/dalpha = α_sigmoid * (1 - α_sigmoid)
        self.dalpha = dalpha_sigmoid * alpha_sigmoid * (1 - alpha_sigmoid)
        
        # 3. 反向传播通过 Mamba 和 GRU 分支
        dx_mamba = self.mamba.backward(dout_mamba, cache['cache_mamba'])
        dx_gru = self.min_gru.backward(dout_gru, cache['cache_gru'])
        
        # 4. 合并两个分支的输入梯度
        # 由于输入是共享的，需要将两个分支的梯度相加
        dx_combined = dx_mamba + dx_gru
        
        return dx_combined
    
    def update_parameters(self, learning_rate):
        """
        使用 SGD 更新参数
        
        参数:
            learning_rate: float, 学习率
        """
        # 更新融合权重
        self.alpha -= learning_rate * self.dalpha
        
        # 更新输出层
        self.W_out -= learning_rate * self.dW_out
        self.b_out -= learning_rate * self.db_out
        
        # 更新子模块
        self.mamba.update_parameters(learning_rate)
        self.min_gru.update_parameters(learning_rate)
    
    def get_parameters(self):
        """
        获取所有参数
        
        返回:
            params: dict, 包含所有参数的字典
        """
        params = {
            'alpha': self.alpha,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'mamba_W_in': self.mamba.W_in,
            'mamba_b_in': self.mamba.b_in,
            'mamba_B': self.mamba.B,
            'mamba_C': self.mamba.C,
            'mamba_D': self.mamba.D,
            'mamba_W_out': self.mamba.W_out,
            'mamba_b_out': self.mamba.b_out,
            'gru_W_z': self.min_gru.W_z,
            'gru_U_z': self.min_gru.U_z,
            'gru_b_z': self.min_gru.b_z,
            'gru_W_r': self.min_gru.W_r,
            'gru_U_r': self.min_gru.U_r,
            'gru_b_r': self.min_gru.b_r,
            'gru_W_h': self.min_gru.W_h,
            'gru_U_h': self.min_gru.U_h,
            'gru_b_h': self.min_gru.b_h,
            'gru_W_y': self.min_gru.W_y,
            'gru_b_y': self.min_gru.b_y
        }
        return params
    
    def get_gradients(self):
        """
        获取所有梯度
        
        返回:
            grads: dict, 包含所有梯度的字典
        """
        grads = {
            'alpha': self.dalpha,
            'W_out': self.dW_out,
            'b_out': self.db_out,
            'mamba_W_in': self.mamba.dW_in,
            'mamba_b_in': self.mamba.db_in,
            'mamba_B': self.mamba.dB,
            'mamba_C': self.mamba.dC,
            'mamba_D': self.mamba.dD,
            'mamba_W_out': self.mamba.dW_out,
            'mamba_b_out': self.mamba.db_out,
            'gru_W_z': self.min_gru.dW_z,
            'gru_U_z': self.min_gru.dU_z,
            'gru_b_z': self.min_gru.db_z,
            'gru_W_r': self.min_gru.dW_r,
            'gru_U_r': self.min_gru.dU_r,
            'gru_b_r': self.min_gru.db_r,
            'gru_W_h': self.min_gru.dW_h,
            'gru_U_h': self.min_gru.dU_h,
            'gru_b_h': self.min_gru.db_h,
            'gru_W_y': self.min_gru.dW_y,
            'gru_b_y': self.min_gru.db_y
        }
        return grads
    
    def set_parameters(self, params):
        """
        设置所有参数
        
        参数:
            params: dict, 包含所有参数的字典
        """
        self.alpha = params['alpha']
        self.W_out = params['W_out']
        self.b_out = params['b_out']
        self.mamba.W_in = params['mamba_W_in']
        self.mamba.b_in = params['mamba_b_in']
        self.mamba.B = params['mamba_B']
        self.mamba.C = params['mamba_C']
        self.mamba.D = params['mamba_D']
        self.mamba.W_out = params['mamba_W_out']
        self.mamba.b_out = params['mamba_b_out']
        self.min_gru.W_z = params['gru_W_z']
        self.min_gru.U_z = params['gru_U_z']
        self.min_gru.b_z = params['gru_b_z']
        self.min_gru.W_r = params['gru_W_r']
        self.min_gru.U_r = params['gru_U_r']
        self.min_gru.b_r = params['gru_b_r']
        self.min_gru.W_h = params['gru_W_h']
        self.min_gru.U_h = params['gru_U_h']
        self.min_gru.b_h = params['gru_b_h']
        self.min_gru.W_y = params['gru_W_y']
        self.min_gru.b_y = params['gru_b_y']
    
    def compute_loss(self, y_pred, y_true):
        """
        计算均方误差损失 (MSE)
        
        参数:
            y_pred: numpy array, shape=(output_size, batch_size)
            y_true: numpy array, shape=(output_size, batch_size)
        
        返回:
            loss: float, MSE 损失值
            dy: numpy array, 损失对输出的梯度
        """
        batch_size = y_pred.shape[1]
        diff = y_pred - y_true
        loss = np.mean(diff ** 2)
        dy = 2 * diff / batch_size
        return loss, dy
    
    def get_fusion_weight(self):
        """
        返回当前的融合权重 (sigmoid 后的值)
        
        返回:
            float or array, 融合权重的值
        """
        return self.sigmoid(self.alpha)


if __name__ == "__main__":
    # 测试 Mamba-GRU 混合模型
    print("=" * 70)
    print("Mamba-GRU 混合模型测试")
    print("=" * 70)
    
    # 初始化模型
    model = MambaGRU(
        input_size=1,
        hidden_size=32,
        output_size=1,
        state_size=64,
        use_vector_alpha=False  # 使用标量 alpha
    )
    
    # 生成测试数据
    seq_len = 10
    batch_size = 4
    X_test = np.random.randn(seq_len, 1, batch_size)
    y_test = np.random.randn(1, batch_size)
    
    print(f"\n输入形状: {X_test.shape}")
    print(f"目标形状: {y_test.shape}")
    
    # 前向传播
    print("\n1. 前向传播...")
    y_pred, cache = model.forward(X_test)
    print(f"   输出形状: {y_pred.shape}")
    print(f"   Mamba 输出形状: {cache['out_mamba'].shape}")
    print(f"   GRU 输出形状: {cache['out_gru'].shape}")
    print(f"   融合权重 α: {np.mean(cache['alpha_sigmoid']):.4f}")
    
    # 计算损失
    print("\n2. 计算损失...")
    loss, dy = model.compute_loss(y_pred, y_test)
    print(f"   MSE 损失: {loss:.6f}")
    
    # 反向传播
    print("\n3. 反向传播...")
    dx = model.backward(dy, cache)
    print(f"   输入梯度形状: {dx.shape}")
    print(f"   alpha 梯度: {model.dalpha}")
    
    # 更新参数
    print("\n4. 更新参数...")
    alpha_before = model.alpha.copy()
    model.update_parameters(learning_rate=0.01)
    alpha_after = model.alpha.copy()
    print(f"   alpha 更新前: {alpha_before.flatten()}")
    print(f"   alpha 更新后: {alpha_after.flatten()}")
    print(f"   alpha 变化: {(alpha_after - alpha_before).flatten()}")
    
    # 测试多个 epoch
    print("\n5. 测试多个训练步骤 (观察 alpha 变化)...")
    for epoch in range(5):
        y_pred, cache = model.forward(X_test)
        loss, dy = model.compute_loss(y_pred, y_test)
        dx = model.backward(dy, cache)
        model.update_parameters(learning_rate=0.01)
        fusion_weight = model.get_fusion_weight()
        print(f"   Epoch {epoch + 1}: Loss = {loss:.6f}, α = {np.mean(fusion_weight):.4f}")
    
    print("\n" + "=" * 70)
    print("Mamba-GRU 混合模型测试成功!")
    print("=" * 70)
