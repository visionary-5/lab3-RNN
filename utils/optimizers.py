"""
优化器模块
实现 Adam、SGD 等优化器的纯 NumPy 实现
"""
import numpy as np


class AdamOptimizer:
    """
    Adam 优化器 (Adaptive Moment Estimation)
    
    数学公式:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # 一阶动量
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        # 二阶动量
        m_hat = m_t / (1 - beta1^t)                        # 偏差修正的一阶动量
        v_hat = v_t / (1 - beta2^t)                        # 偏差修正的二阶动量
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    参数:
        learning_rate: float, 学习率
        beta1: float, 一阶动量衰减系数 (默认 0.9)
        beta2: float, 二阶动量衰减系数 (默认 0.999)
        epsilon: float, 数值稳定性常数 (默认 1e-8)
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # 时间步计数器
        self.m = {}  # 一阶动量字典 {param_name: m_value}
        self.v = {}  # 二阶动量字典 {param_name: v_value}
    
    def update(self, params, grads):
        """
        使用 Adam 算法更新参数
        
        参数:
            params: dict, {param_name: param_value}
            grads: dict, {param_name: grad_value}
        
        返回:
            updated_params: dict, 更新后的参数
        """
        self.t += 1
        updated_params = {}
        
        for param_name, param in params.items():
            grad = grads[param_name]
            
            # 初始化动量
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param)
                self.v[param_name] = np.zeros_like(param)
            
            # 更新一阶动量 (带梯度的指数移动平均)
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            
            # 更新二阶动量 (梯度平方的指数移动平均)
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            updated_params[param_name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params
    
    def set_learning_rate(self, new_lr):
        """设置新的学习率"""
        self.learning_rate = new_lr


class SGDOptimizer:
    """
    SGD 优化器 (Stochastic Gradient Descent)
    支持动量 (Momentum)
    
    数学公式:
        v_t = momentum * v_{t-1} + lr * g_t
        theta_t = theta_{t-1} - v_t
    
    参数:
        learning_rate: float, 学习率
        momentum: float, 动量系数 (0 表示不使用动量)
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}  # 速度字典
    
    def update(self, params, grads):
        """
        使用 SGD (with momentum) 更新参数
        
        参数:
            params: dict, {param_name: param_value}
            grads: dict, {param_name: grad_value}
        
        返回:
            updated_params: dict, 更新后的参数
        """
        updated_params = {}
        
        for param_name, param in params.items():
            grad = grads[param_name]
            
            # 初始化速度
            if param_name not in self.v:
                self.v[param_name] = np.zeros_like(param)
            
            # 更新速度
            self.v[param_name] = self.momentum * self.v[param_name] + self.learning_rate * grad
            
            # 参数更新
            updated_params[param_name] = param - self.v[param_name]
        
        return updated_params
    
    def set_learning_rate(self, new_lr):
        """设置新的学习率"""
        self.learning_rate = new_lr


if __name__ == "__main__":
    # 测试 Adam 优化器
    print("测试 Adam 优化器:")
    adam = AdamOptimizer(learning_rate=0.001)
    
    # 模拟参数和梯度
    params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]]), 'b': np.array([0.5])}
    grads = {'W': np.array([[0.1, 0.2], [0.3, 0.4]]), 'b': np.array([0.05])}
    
    print("初始参数:")
    print(f"W:\n{params['W']}")
    print(f"b: {params['b']}")
    
    # 更新参数
    updated = adam.update(params, grads)
    print("\n更新后参数:")
    print(f"W:\n{updated['W']}")
    print(f"b: {updated['b']}")
    
    print("\nAdam 优化器测试成功!")
