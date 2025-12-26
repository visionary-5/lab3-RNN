"""
Mamba 模型实现
使用纯 NumPy 实现基于状态空间模型 (SSM) 的 Mamba 架构
"""
import numpy as np


class Mamba:
    """
    Mamba 模型 - 基于选择性状态空间模型 (Selective State Space Model)
    
    核心思想:
        1. 使用状态空间模型 (SSM) 处理序列
        2. 引入选择性机制 (Selection Mechanism) 动态调整状态转移
        3. 使用门控机制控制信息流动
    
    状态空间模型公式:
        s_t = A * s_{t-1} + B * gate_t  # 状态更新
        y_t = C @ s_t + D * x_t          # 输出计算
    
    其中 gate_t 通过 SiLU 激活函数从输入计算得到
    """
    
    def __init__(self, input_size, hidden_size, output_size, state_size=64, seed=42):
        """
        初始化 Mamba 模型参数
        
        参数:
            input_size: int, 输入特征维度
            hidden_size: int, 隐藏层维度
            output_size: int, 输出维度
            state_size: int, 状态空间维度
            seed: int, 随机种子
        """
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size
        
        # 输入投影层 (Xavier 初始化)
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W_in = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.b_in = np.zeros((hidden_size, 1))
        
        # 状态空间模型参数
        # A: 状态衰减矩阵 (固定为负值以确保稳定性)
        self.A = -np.ones((state_size, 1)) * 1.0
        
        # B: 输入到状态的映射矩阵
        self.B = np.random.randn(state_size, hidden_size) * 0.01
        
        # C: 状态到输出的映射矩阵
        self.C = np.random.randn(hidden_size, state_size) * 0.01
        
        # D: 跳跃连接权重
        self.D = np.random.randn(hidden_size, hidden_size) * 0.01
        
        # 输出层
        self.W_out = np.random.randn(output_size, hidden_size) * 0.01
        self.b_out = np.zeros((output_size, 1))
        
        # 重置梯度
        self.reset_gradients()
    
    def reset_gradients(self):
        """重置所有梯度为零"""
        self.dW_in = np.zeros_like(self.W_in)
        self.db_in = np.zeros_like(self.b_in)
        self.dB = np.zeros_like(self.B)
        self.dC = np.zeros_like(self.C)
        self.dD = np.zeros_like(self.D)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)
    
    def silu(self, x):
        """
        SiLU (Sigmoid Linear Unit) 激活函数
        SiLU(x) = x * sigmoid(x)
        """
        sigmoid_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return x * sigmoid_x
    
    def silu_derivative(self, x):
        """
        SiLU 的导数
        d/dx[SiLU(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """
        sigmoid_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return sigmoid_x * (1 + x * (1 - sigmoid_x))
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: numpy array, shape=(seq_len, input_size, batch_size)
        
        返回:
            y_pred: numpy array, shape=(output_size, batch_size), 最终输出
            cache: dict, 包含前向传播过程中的所有中间变量
        """
        seq_len, input_size, batch_size = X.shape
        
        # 初始化状态向量
        s = np.zeros((self.state_size, batch_size))
        
        # 存储中间变量
        cache = {
            'X': X,
            's_history': [s.copy()],  # 每个时间步的状态
            'x_proj_history': [],     # 投影后的输入
            'gate_history': [],       # 门控值
            'x_proj_pre': [],         # 投影前的值 (用于反向传播)
            'y_history': []           # 每个时间步的隐藏输出
        }
        
        # 遍历序列的每个时间步
        for t in range(seq_len):
            x_t = X[t]  # shape: (input_size, batch_size)
            
            # 1. 输入投影: 将输入映射到隐藏空间
            x_proj_pre = self.W_in @ x_t + self.b_in
            x_proj = self.silu(x_proj_pre)  # shape: (hidden_size, batch_size)
            
            # 2. 计算选通门 (gate)
            gate = x_proj  # 使用投影后的值作为门控信号
            
            # 3. 状态更新 (离散化的 SSM)
            # s_t = exp(A) * s_{t-1} + B * gate_t
            # 这里使用简化版本: s_t = (1 + A) * s_{t-1} + B * gate_t
            exp_A = np.exp(self.A)  # shape: (state_size, 1)
            s = s * exp_A + (self.B @ gate)  # 广播运算
            
            # 4. 计算隐藏输出
            # y_t = C @ s_t + D * x_proj_t
            y_t = self.C @ s + self.D @ x_proj
            
            # 保存中间变量
            cache['s_history'].append(s.copy())
            cache['x_proj_history'].append(x_proj)
            cache['x_proj_pre'].append(x_proj_pre)
            cache['gate_history'].append(gate)
            cache['y_history'].append(y_t)
        
        # 5. 输出层: 使用最后一个时间步的隐藏输出
        y_last = cache['y_history'][-1]
        y_pred = self.W_out @ y_last + self.b_out
        cache['y_pred'] = y_pred
        cache['y_last'] = y_last
        
        return y_pred, cache
    
    def backward(self, dy, cache):
        """
        反向传播
        
        参数:
            dy: numpy array, shape=(output_size, batch_size), 输出层的梯度
            cache: dict, 前向传播时保存的中间变量
        
        返回:
            dx: numpy array, shape=(seq_len, input_size, batch_size), 输入的梯度
        """
        X = cache['X']
        seq_len, input_size, batch_size = X.shape
        
        # 重置梯度
        self.reset_gradients()
        
        # 初始化梯度
        dx = np.zeros_like(X)
        
        # 1. 输出层梯度
        y_last = cache['y_last']
        self.dW_out += dy @ y_last.T
        self.db_out += np.sum(dy, axis=1, keepdims=True)
        
        # 传递到最后一个隐藏层的梯度
        dy_last = self.W_out.T @ dy
        
        # 2. 初始化状态梯度
        ds_next = np.zeros((self.state_size, batch_size))
        
        # 3. 反向传播通过时间
        for t in reversed(range(seq_len)):
            x_t = X[t]  # (input_size, batch_size)
            x_proj = cache['x_proj_history'][t]
            x_proj_pre = cache['x_proj_pre'][t]
            gate = cache['gate_history'][t]
            s_prev = cache['s_history'][t]
            s_curr = cache['s_history'][t+1]
            
            # 当前时间步的隐藏输出梯度
            if t == seq_len - 1:
                dy_t = dy_last
            else:
                dy_t = np.zeros_like(dy_last)
            
            # 加上来自下一个时间步的状态梯度贡献
            dy_t += self.C.T @ ds_next
            
            # 反向通过输出公式: y_t = C @ s + D @ x_proj
            ds = self.C.T @ dy_t  # 状态的梯度
            dx_proj = self.D.T @ dy_t  # x_proj 的梯度
            
            # 累积 C 和 D 的梯度
            self.dC += dy_t @ s_curr.T
            self.dD += dy_t @ x_proj.T
            
            # 反向通过状态更新: s = s_prev * exp(A) + B @ gate
            exp_A = np.exp(self.A)
            ds_next = ds * exp_A  # 传递到前一个时间步的状态梯度
            
            # B 的梯度
            dgate = self.B.T @ ds
            self.dB += ds @ gate.T
            
            # 门控的梯度 (gate = x_proj)
            dx_proj += dgate
            
            # 反向通过 SiLU 激活
            dx_proj_pre = dx_proj * self.silu_derivative(x_proj_pre)
            
            # 输入投影层的梯度
            self.dW_in += dx_proj_pre @ x_t.T
            self.db_in += np.sum(dx_proj_pre, axis=1, keepdims=True)
            
            # 输入的梯度
            dx[t] = self.W_in.T @ dx_proj_pre
        
        return dx
    
    def update_parameters(self, learning_rate):
        """
        使用 SGD 更新参数
        
        参数:
            learning_rate: float, 学习率
        """
        self.W_in -= learning_rate * self.dW_in
        self.b_in -= learning_rate * self.db_in
        self.B -= learning_rate * self.dB
        self.C -= learning_rate * self.dC
        self.D -= learning_rate * self.dD
        self.W_out -= learning_rate * self.dW_out
        self.b_out -= learning_rate * self.db_out
        
        # 注意: A 参数通常保持固定，不进行更新
    
    def get_parameters(self):
        """
        获取所有参数
        
        返回:
            params: dict, 包含所有参数的字典
        """
        params = {
            'W_in': self.W_in,
            'b_in': self.b_in,
            'B': self.B,
            'C': self.C,
            'D': self.D,
            'W_out': self.W_out,
            'b_out': self.b_out
        }
        return params
    
    def get_gradients(self):
        """
        获取所有梯度
        
        返回:
            grads: dict, 包含所有梯度的字典
        """
        grads = {
            'W_in': self.dW_in,
            'b_in': self.db_in,
            'B': self.dB,
            'C': self.dC,
            'D': self.dD,
            'W_out': self.dW_out,
            'b_out': self.db_out
        }
        return grads
    
    def set_parameters(self, params):
        """
        设置所有参数
        
        参数:
            params: dict, 包含所有参数的字典
        """
        self.W_in = params['W_in']
        self.b_in = params['b_in']
        self.B = params['B']
        self.C = params['C']
        self.D = params['D']
        self.W_out = params['W_out']
        self.b_out = params['b_out']
    
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


if __name__ == "__main__":
    # 简单测试
    print("Mamba 模型测试:")
    model = Mamba(input_size=1, hidden_size=32, output_size=1, state_size=64)
    
    # 生成测试数据
    seq_len = 10
    batch_size = 4
    X_test = np.random.randn(seq_len, 1, batch_size)
    y_test = np.random.randn(1, batch_size)
    
    # 前向传播
    y_pred, cache = model.forward(X_test)
    print(f"输入形状: {X_test.shape}")
    print(f"输出形状: {y_pred.shape}")
    
    # 计算损失
    loss, dy = model.compute_loss(y_pred, y_test)
    print(f"损失值: {loss:.6f}")
    
    # 反向传播
    dx = model.backward(dy, cache)
    print(f"输入梯度形状: {dx.shape}")
    
    # 更新参数
    model.update_parameters(learning_rate=0.01)
    print("参数更新完成!")
    
    print("\nMamba 模型测试成功!")
