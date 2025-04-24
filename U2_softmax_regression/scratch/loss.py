import torch

def cross_entropy(y_hat, y):
    """定义交叉熵损失函数"""
    return -torch.log(y_hat[range(len(y_hat)), y])