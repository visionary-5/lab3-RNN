"""
学习率调度器模块
实现学习率衰减策略
"""
import numpy as np


class StepLR:
    """
    阶梯式学习率衰减
    
    每隔 step_size 个 epoch，学习率乘以 gamma
    
    公式:
        lr_t = lr_0 * gamma^(t // step_size)
    
    参数:
        optimizer: 优化器对象
        step_size: int, 学习率衰减的间隔 epoch 数
        gamma: float, 学习率衰减系数
    """
    
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = optimizer.learning_rate
        self.epoch = 0
    
    def step(self):
        """在每个 epoch 结束时调用，更新学习率"""
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            new_lr = self.optimizer.learning_rate * self.gamma
            self.optimizer.set_learning_rate(new_lr)
            print(f"  [LR Scheduler] Epoch {self.epoch}: 学习率衰减至 {new_lr:.6f}")
    
    def get_last_lr(self):
        """返回当前学习率"""
        return self.optimizer.learning_rate


class CosineAnnealingLR:
    """
    余弦退火学习率调度器
    
    学习率按照余弦函数曲线从初始值平滑衰减到最小值
    
    公式:
        lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))
    
    参数:
        optimizer: 优化器对象
        T_max: int, 半个余弦周期的 epoch 数
        eta_min: float, 最小学习率
    """
    
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = optimizer.learning_rate
        self.epoch = 0
    
    def step(self):
        """在每个 epoch 结束时调用，更新学习率"""
        self.epoch += 1
        # 余弦退火公式
        new_lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                 (1 + np.cos(np.pi * self.epoch / self.T_max))
        self.optimizer.set_learning_rate(new_lr)
    
    def get_last_lr(self):
        """返回当前学习率"""
        return self.optimizer.learning_rate


class ExponentialLR:
    """
    指数衰减学习率调度器
    
    公式:
        lr_t = lr_0 * gamma^t
    
    参数:
        optimizer: 优化器对象
        gamma: float, 学习率衰减系数 (每个 epoch 乘以 gamma)
    """
    
    def __init__(self, optimizer, gamma=0.95):
        self.optimizer = optimizer
        self.gamma = gamma
        self.epoch = 0
    
    def step(self):
        """在每个 epoch 结束时调用，更新学习率"""
        self.epoch += 1
        new_lr = self.optimizer.learning_rate * self.gamma
        self.optimizer.set_learning_rate(new_lr)
    
    def get_last_lr(self):
        """返回当前学习率"""
        return self.optimizer.learning_rate


if __name__ == "__main__":
    from optimizers import AdamOptimizer
    
    print("测试学习率调度器:")
    
    # 测试 StepLR
    print("\n1. StepLR 测试:")
    optimizer = AdamOptimizer(learning_rate=0.01)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    
    for epoch in range(10):
        print(f"Epoch {epoch + 1}: lr = {scheduler.get_last_lr():.6f}")
        scheduler.step()
    
    # 测试 CosineAnnealingLR
    print("\n2. CosineAnnealingLR 测试:")
    optimizer2 = AdamOptimizer(learning_rate=0.01)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=10, eta_min=0.0001)
    
    for epoch in range(10):
        print(f"Epoch {epoch + 1}: lr = {scheduler2.get_last_lr():.6f}")
        scheduler2.step()
    
    print("\n学习率调度器测试成功!")
