import torch
from d2l import torch as d2l

from utils import Accumulator, evaluate_accuracy, accuracy

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # net.train()  # 移除这行，因为 net 是一个函数，不是 torch.nn.Module
    metric = Accumulator(3)  # 训练损失、训练准确度、样本数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.sum().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(l.sum(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            yscale='log', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    print(f'train loss {train_loss:f}, train acc {train_acc:f}, test acc {test_acc:f}')