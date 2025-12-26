"""
MinGRU 模型实现
使用纯 NumPy 实现 Minimal Gated Recurrent Unit
"""
import numpy as np


class MinGRU:
    """
    MinGRU (Minimal Gated Recurrent Unit) 模型
    
    一个简化版的 GRU，包含更新门和重置门机制
    公式:
        z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)  # 更新门
        r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1} + b_r)  # 重置门
        h_tilde = tanh(W_h @ x_t + U_h @ (r_t * h_{t-1}) + b_h)  # 候选隐藏状态
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde  # 最终隐藏状态
    """
    
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        """
        初始化 MinGRU 模型参数
        
        参数:
            input_size: int, 输入特征维度
            hidden_size: int, 隐藏层维度
            output_size: int, 输出维度
            seed: int, 随机种子
        """
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重矩阵 (使用小的随机值)
        # 更新门参数
        self.W_z = np.random.randn(hidden_size, input_size) * 0.01
        self.U_z = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros((hidden_size, 1))
        
        # 重置门参数
        self.W_r = np.random.randn(hidden_size, input_size) * 0.01
        self.U_r = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_r = np.zeros((hidden_size, 1))
        
        # 候选隐藏状态参数
        self.W_h = np.random.randn(hidden_size, input_size) * 0.01
        self.U_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        
        # 输出层参数
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        # 用于存储梯度的变量
        self.reset_gradients()
    
    def reset_gradients(self):
        """重置所有梯度为零"""
        self.dW_z = np.zeros_like(self.W_z)
        self.dU_z = np.zeros_like(self.U_z)
        self.db_z = np.zeros_like(self.b_z)
        
        self.dW_r = np.zeros_like(self.W_r)
        self.dU_r = np.zeros_like(self.U_r)
        self.db_r = np.zeros_like(self.b_r)
        
        self.dW_h = np.zeros_like(self.W_h)
        self.dU_h = np.zeros_like(self.U_h)
        self.db_h = np.zeros_like(self.b_h)
        
        self.dW_y = np.zeros_like(self.W_y)
        self.db_y = np.zeros_like(self.b_y)
    
    def sigmoid(self, x):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Tanh 激活函数"""
        return np.tanh(x)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: numpy array, shape=(seq_len, input_size, batch_size)
        
        返回:
            y_pred: numpy array, shape=(output_size, batch_size), 最终输出
            cache: dict, 包含前向传播过程中的所有中间变量，用于反向传播
        """
        seq_len, input_size, batch_size = X.shape
        
        # 初始化隐藏状态
        h = np.zeros((self.hidden_size, batch_size))
        
        # 存储每个时间步的中间变量
        cache = {
            'X': X,
            'h': [h],  # 存储所有时间步的隐藏状态
            'z': [],   # 更新门
            'r': [],   # 重置门
            'h_tilde': [],  # 候选隐藏状态
            'z_pre': [],  # 更新门激活前的值
            'r_pre': [],  # 重置门激活前的值
            'h_tilde_pre': []  # 候选隐藏状态激活前的值
        }
        
        # 遍历序列的每个时间步
        for t in range(seq_len):
            x_t = X[t]  # shape: (input_size, batch_size)
            
            # 1. 计算更新门 z_t
            z_pre = self.W_z @ x_t + self.U_z @ h + self.b_z
            z_t = self.sigmoid(z_pre)
            
            # 2. 计算重置门 r_t
            r_pre = self.W_r @ x_t + self.U_r @ h + self.b_r
            r_t = self.sigmoid(r_pre)
            
            # 3. 计算候选隐藏状态 h_tilde
            h_tilde_pre = self.W_h @ x_t + self.U_h @ (r_t * h) + self.b_h
            h_tilde = self.tanh(h_tilde_pre)
            
            # 4. 更新隐藏状态
            h = (1 - z_t) * h + z_t * h_tilde
            
            # 保存中间变量
            cache['h'].append(h.copy())
            cache['z'].append(z_t)
            cache['r'].append(r_t)
            cache['h_tilde'].append(h_tilde)
            cache['z_pre'].append(z_pre)
            cache['r_pre'].append(r_pre)
            cache['h_tilde_pre'].append(h_tilde_pre)
        
        # 5. 计算输出 (使用最后一个时间步的隐藏状态)
        y_pred = self.W_y @ h + self.b_y
        cache['y_pred'] = y_pred
        
        return y_pred, cache
    
    def backward(self, dy, cache):
        """
        反向传播 (BPTT - Backpropagation Through Time)
        
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
        dh_next = np.zeros((self.hidden_size, batch_size))
        dx = np.zeros_like(X)
        
        # 1. 输出层梯度
        self.dW_y += dy @ cache['h'][-1].T
        self.db_y += np.sum(dy, axis=1, keepdims=True)
        dh_next += self.W_y.T @ dy
        
        # 2. 反向传播通过时间
        for t in reversed(range(seq_len)):
            x_t = X[t]  # (input_size, batch_size)
            h_prev = cache['h'][t]  # (hidden_size, batch_size)
            h_curr = cache['h'][t+1]
            
            z_t = cache['z'][t]
            r_t = cache['r'][t]
            h_tilde = cache['h_tilde'][t]
            
            # 当前时间步的隐藏状态梯度
            dh = dh_next
            
            # 反向通过隐藏状态更新公式: h = (1-z) * h_prev + z * h_tilde
            dz = dh * (h_tilde - h_prev)  # 更新门的梯度
            dh_tilde = dh * z_t  # 候选隐藏状态的梯度
            dh_prev = dh * (1 - z_t)  # 传递到前一个时间步的梯度
            
            # 反向通过 tanh: h_tilde = tanh(h_tilde_pre)
            dh_tilde_pre = dh_tilde * (1 - h_tilde ** 2)
            
            # 候选隐藏状态的参数梯度
            self.dW_h += dh_tilde_pre @ x_t.T
            self.dU_h += dh_tilde_pre @ (r_t * h_prev).T
            self.db_h += np.sum(dh_tilde_pre, axis=1, keepdims=True)
            
            # 反向通过重置门
            dr = (self.U_h.T @ dh_tilde_pre) * h_prev
            dh_prev += (self.U_h.T @ dh_tilde_pre) * r_t
            
            # 反向通过 sigmoid: r_t = sigmoid(r_pre)
            dr_pre = dr * r_t * (1 - r_t)
            
            # 重置门的参数梯度
            self.dW_r += dr_pre @ x_t.T
            self.dU_r += dr_pre @ h_prev.T
            self.db_r += np.sum(dr_pre, axis=1, keepdims=True)
            
            # 反向通过 sigmoid: z_t = sigmoid(z_pre)
            dz_pre = dz * z_t * (1 - z_t)
            
            # 更新门的参数梯度
            self.dW_z += dz_pre @ x_t.T
            self.dU_z += dz_pre @ h_prev.T
            self.db_z += np.sum(dz_pre, axis=1, keepdims=True)
            
            # 输入的梯度
            dx_t = self.W_z.T @ dz_pre + self.W_r.T @ dr_pre + self.W_h.T @ dh_tilde_pre
            dx[t] = dx_t
            
            # 传递到前一个时间步的隐藏状态梯度
            dh_next = dh_prev + self.U_z.T @ dz_pre + self.U_r.T @ dr_pre
        
        return dx
    
    def update_parameters(self, learning_rate):
        """
        使用 SGD 更新参数
        
        参数:
            learning_rate: float, 学习率
        """
        # 更新门参数
        self.W_z -= learning_rate * self.dW_z
        self.U_z -= learning_rate * self.dU_z
        self.b_z -= learning_rate * self.db_z
        
        # 重置门参数
        self.W_r -= learning_rate * self.dW_r
        self.U_r -= learning_rate * self.dU_r
        self.b_r -= learning_rate * self.db_r
        
        # 候选隐藏状态参数
        self.W_h -= learning_rate * self.dW_h
        self.U_h -= learning_rate * self.dU_h
        self.b_h -= learning_rate * self.db_h
        
        # 输出层参数
        self.W_y -= learning_rate * self.dW_y
        self.b_y -= learning_rate * self.db_y
    
    def get_parameters(self):
        """
        获取所有参数
        
        返回:
            params: dict, 包含所有参数的字典
        """
        params = {
            'W_z': self.W_z,
            'U_z': self.U_z,
            'b_z': self.b_z,
            'W_r': self.W_r,
            'U_r': self.U_r,
            'b_r': self.b_r,
            'W_h': self.W_h,
            'U_h': self.U_h,
            'b_h': self.b_h,
            'W_y': self.W_y,
            'b_y': self.b_y
        }
        return params
    
    def get_gradients(self):
        """
        获取所有梯度
        
        返回:
            grads: dict, 包含所有梯度的字典
        """
        grads = {
            'W_z': self.dW_z,
            'U_z': self.dU_z,
            'b_z': self.db_z,
            'W_r': self.dW_r,
            'U_r': self.dU_r,
            'b_r': self.db_r,
            'W_h': self.dW_h,
            'U_h': self.dU_h,
            'b_h': self.db_h,
            'W_y': self.dW_y,
            'b_y': self.db_y
        }
        return grads
    
    def set_parameters(self, params):
        """
        设置所有参数
        
        参数:
            params: dict, 包含所有参数的字典
        """
        self.W_z = params['W_z']
        self.U_z = params['U_z']
        self.b_z = params['b_z']
        self.W_r = params['W_r']
        self.U_r = params['U_r']
        self.b_r = params['b_r']
        self.W_h = params['W_h']
        self.U_h = params['U_h']
        self.b_h = params['b_h']
        self.W_y = params['W_y']
        self.b_y = params['b_y']
    
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
    print("MinGRU 模型测试:")
    model = MinGRU(input_size=1, hidden_size=32, output_size=1)
    
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
    
    print("\nMinGRU 模型测试成功!")
