import torch

def softmax(X):
    """定义softmax操作"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X, W, b):
    """定义模型"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)