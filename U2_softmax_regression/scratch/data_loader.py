from d2l import torch as d2l

def load_data(batch_size):
    """
        加载Fashion-MNIST数据集:
        1. 训练集: 60000张28*28的灰度图像
        2. 测试集: 10000张28*28的灰度图像
        3. 每张图像对应一个标签, 标签范围为0-9
        4. 标签对应的图像分别为: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
        5. 每个batch_size包含64张图像
    """
    return d2l.load_data_fashion_mnist(batch_size)