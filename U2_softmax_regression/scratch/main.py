import torch
from data_loader import load_data
from model import net
from loss import cross_entropy
from utils import evaluate_accuracy
from train import train_ch3

# batch_size代表每个小批量的样本数
# train_iter和test_iter分别是训练集和测试集的迭代器
batch_size = 256
train_iter, test_iter = load_data(batch_size) # len(train_iter)= 60000/256=235次

num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

lr = 0.1
num_epochs = 10

# 更新模型参数
def updater(batch_size):
    """使用小批量随机梯度下降更新参数"""
    with torch.no_grad():
        for param in [W, b]:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
train_ch3(lambda X: net(X, W, b), train_iter, test_iter, cross_entropy, num_epochs, updater)