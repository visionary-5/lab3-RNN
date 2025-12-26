"""
正则化模块
实现梯度裁剪、L2 正则化、Dropout 等
"""
import numpy as np


def gradient_clipping(grads, max_norm=5.0):
    """
    梯度裁剪 (Gradient Clipping)
    
    防止梯度爆炸，将梯度的范数限制在 max_norm 以内
    
    公式:
        如果 ||g|| > max_norm:
            g = g * (max_norm / ||g||)
    
    参数:
        grads: dict, {param_name: grad_value}
        max_norm: float, 梯度的最大范数
    
    返回:
        clipped_grads: dict, 裁剪后的梯度
        total_norm: float, 裁剪前的梯度总范数
    """
    # 计算所有梯度的总范数
    total_norm = 0.0
    for grad in grads.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # 计算裁剪系数
    clip_coef = max_norm / (total_norm + 1e-8)
    
    # 如果总范数超过 max_norm，则进行裁剪
    clipped_grads = {}
    if clip_coef < 1.0:
        for param_name, grad in grads.items():
            clipped_grads[param_name] = grad * clip_coef
    else:
        clipped_grads = grads.copy()
    
    return clipped_grads, total_norm


def l2_regularization_loss(params, lambda_reg=0.01):
    """
    计算 L2 正则化损失 (Weight Decay)
    
    公式:
        L2_loss = (lambda / 2) * Σ||W||^2
    
    参数:
        params: dict, {param_name: param_value}
        lambda_reg: float, 正则化系数
    
    返回:
        float, L2 正则化损失
    """
    l2_loss = 0.0
    for param in params.values():
        l2_loss += np.sum(param ** 2)
    return 0.5 * lambda_reg * l2_loss


def l2_regularization_grad(params, lambda_reg=0.01):
    """
    计算 L2 正则化的梯度
    
    公式:
        dL/dW = lambda * W
    
    参数:
        params: dict, {param_name: param_value}
        lambda_reg: float, 正则化系数
    
    返回:
        reg_grads: dict, L2 正则化的梯度
    """
    reg_grads = {}
    for param_name, param in params.items():
        reg_grads[param_name] = lambda_reg * param
    return reg_grads


class Dropout:
    """
    Dropout 正则化
    
    在训练时随机将一部分神经元的输出置为 0，防止过拟合
    在测试时不进行 dropout，但需要将输出乘以 (1 - p) 进行补偿
    
    参数:
        p: float, dropout 概率 (0 到 1 之间)
    """
    
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: numpy array, 输入数据
        
        返回:
            output: numpy array, dropout 后的输出
        """
        if self.training:
            # 生成 dropout mask (伯努利分布)
            self.mask = np.random.binomial(1, 1 - self.p, size=X.shape)
            # 应用 mask 并进行缩放 (inverted dropout)
            return X * self.mask / (1 - self.p)
        else:
            # 测试时不进行 dropout
            return X
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: numpy array, 上游梯度
        
        返回:
            dx: numpy array, 输入的梯度
        """
        if self.training:
            return dout * self.mask / (1 - self.p)
        else:
            return dout
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False


class EarlyStopping:
    """
    Early Stopping (早停法)
    
    当验证集损失在连续 patience 个 epoch 没有改善时停止训练
    
    参数:
        patience: int, 耐心值，连续多少个 epoch 没有改善就停止
        min_delta: float, 最小改善量，小于这个值不认为是改善
        verbose: bool, 是否打印信息
    """
    
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        """
        检查是否应该停止训练
        
        参数:
            val_loss: float, 当前验证集损失
            epoch: int, 当前 epoch
        
        返回:
            bool, 是否应该停止训练
        """
        if self.best_loss is None:
            # 第一次调用
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            # 验证集损失有改善
            if self.verbose:
                print(f"  [Early Stopping] 验证损失改善: {self.best_loss:.6f} -> {val_loss:.6f}")
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            # 验证集损失没有改善
            self.counter += 1
            if self.verbose:
                print(f"  [Early Stopping] 验证损失未改善 ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [Early Stopping] 触发早停！最佳 Epoch: {self.best_epoch}, 最佳验证损失: {self.best_loss:.6f}")
        
        return self.early_stop


if __name__ == "__main__":
    # 测试梯度裁剪
    print("测试梯度裁剪:")
    grads = {
        'W1': np.array([[10.0, 20.0], [30.0, 40.0]]),
        'b1': np.array([5.0])
    }
    print(f"原始梯度: {grads}")
    clipped, norm = gradient_clipping(grads, max_norm=5.0)
    print(f"裁剪后梯度: {clipped}")
    print(f"梯度范数: {norm:.4f}")
    
    # 测试 Dropout
    print("\n测试 Dropout:")
    dropout = Dropout(p=0.5)
    X = np.ones((5, 3))
    print(f"输入:\n{X}")
    dropout.train()
    out = dropout.forward(X)
    print(f"训练模式输出:\n{out}")
    dropout.eval()
    out = dropout.forward(X)
    print(f"评估模式输出:\n{out}")
    
    print("\n正则化模块测试成功!")
