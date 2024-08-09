import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from models.module import Seq2Seq
'''
     1、初式化参数，包括:
          - letter列表，包含了所有可能的字符（包括'S', 'E', '?'以及所有的小写字母）。这个列表用于定义模型可以处理的所有可能的输入字符
          - letter2idx字典，key是letter列表中的字符，value是这些字符在letter列表中的索引
          - seq_data训练数据列表
          - n_step最大长度
          - n_hidden表示隐藏层的神经元数量。在这个代码中，n_hidden被设置为128，意味着在Seq2Seq模型的编码器和解码器的RNN层中，每个隐藏层包含128个神经元
          - n_class表示字符集的大小，也就是模型可以处理的不同字符的数量，即分类问题的种类
          - batch_size定义在训练神经网络时每个批次（batch）中的样本数量
     2、make_data()预处理函数，将输入的seq_data转换为模型可以处理的数据格式（即：文本->向量）
     3、Data.DataLoader(TranslateDataSet(...)) 格式化数据集向量，将数据集封装为DataLoader对象，准备训练
     4、实例化模型、损失函数、优化器:
           - 模型：Seq2Seq(n_class, n_hidden)
           - 损失函数：nn.CrossEntropyLoss()
           - 优化器：torch.optim.Adam(model.parameters(), lr=0.001)
     5、训练模型
     6、测试模型
     
     
     注：
        # S: 解码器输入标志字符
        # E: 解码器输出标志字符
        # ?: 填充字符（如果输入字符长度小于n_step，用？补齐）
'''

# 设置设备（判断是否有可用的GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 词典
letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']  # 词典
# 字典（key：letter各字符，value：对应索引）
letter2idx = {n: i for i, n in enumerate(letter)}
# 训练数据
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

# Seq2Seq Parameter
# 训练集最大长度
n_step = max([max(len(i), len(j)) for i, j in seq_data])
# 隐藏层的神经元数量
n_hidden = 128
# 字符集的大小，表示匹配字符的种类
n_class = len(letter2idx)
# 每个批次（batch）中的样本数量
batch_size = 3

def make_data(seq_data):
    enc_input_all, dec_input_all, dec_output_all = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + '?' * (n_step - len(seq[i]))  # 填充?，使得长度相同
            print("seq[i]",seq[i])

        enc_input = [letter2idx[n] for n in (seq[0] + 'E')]  # ['m', 'a', 'n', '?', '?', 'E']
        dec_input = [letter2idx[n] for n in ('S' + seq[1])]  # ['S', 'w', 'o', 'm', 'e', 'n']
        # todo
        dec_output = [letter2idx[n] for n in (seq[1] + 'E')]  # ['w', 'o', 'm', 'e', 'n', 'E']

        enc_input_all.append(np.eye(n_class)[enc_input])
        dec_input_all.append(np.eye(n_class)[dec_input])
        dec_output_all.append(dec_output)  # not one-hot
    # make tensor
    return torch.Tensor(enc_input_all), torch.Tensor(dec_input_all), torch.LongTensor(dec_output_all)

'''
enc_input_all: [6, n_step+1 (because of 'E'), n_class]
dec_input_all: [6, n_step+1 (because of 'S'), n_class]
dec_output_all: [6, n_step+1 (because of 'E')]
'''
# 数据预处理
enc_input_all, dec_input_all, dec_output_all = make_data(seq_data)


class TranslateDataSet(Data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all

    def __len__(self):  # return dataset size
        return len(self.enc_input_all)

    def __getitem__(self, idx):
        return self.enc_input_all[idx], self.dec_input_all[idx], self.dec_output_all[idx]

# 数据封装为DataLoader对象，以便迭代训练
# shuffle：用于控制是否在每个训练周期（epoch）开始时对数据进行随机重排。
loader = Data.DataLoader(TranslateDataSet(enc_input_all, dec_input_all, dec_output_all), batch_size, True)

# 初始化模型
model = Seq2Seq(n_class, n_hidden).to(device)  # 实例化模型到cpu或gpu上
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 训练模型（5000*2*3）
for epoch in range(5000):
    for enc_input_batch, dec_input_batch, dec_output_batch in loader:
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        # 隐藏状态初始化
        h_0 = torch.zeros(1, batch_size, n_hidden).to(device)

        (enc_input_batch, dec_intput_batch, dec_output_batch) = (
            enc_input_batch.to(device), dec_input_batch.to(device), dec_output_batch.to(device))
        # enc_input_batch : [batch_size, n_step+1, n_class]
        # dec_intput_batch : [batch_size, n_step+1, n_class]
        # dec_output_batch : [batch_size, n_step+1], not one-hot
        pred = model(enc_input_batch, h_0, dec_intput_batch) # 预测结果
        # pred : [n_step+1, batch_size, n_class]
        pred = pred.transpose(0, 1)  # [batch_size, n_step+1(=6), n_class] 将预测结果转置，以便于后续的损失计算
        loss = 0
        for i in range(len(dec_output_batch)):
            # pred[i] : [n_step+1, n_class]
            # dec_output_batch[i] : [n_step+1]
            loss += criterion(pred[i], dec_output_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward() # 不太理解
        optimizer.step()


# Test
def translate(word):
    enc_input, dec_input, _ = make_data([[word, '?' * n_step]])
    enc_input, dec_input = enc_input.to(device), dec_input.to(device)
    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    hidden = torch.zeros(1, 1, n_hidden).to(device)
    output = model(enc_input, hidden, dec_input)
    # output : [n_step+1, batch_size, n_class]

    predict = output.data.max(2, keepdim=True)[1]  # select n_class dimension
    decoded = [letter[i] for i in predict]
    translated = ''.join(decoded[:decoded.index('E')])

    return translated.replace('?', '')


print('test')
# print('man ->', translate('man'))
# print('mans ->', translate('mans'))
# print('king ->', translate('king'))
# print('black ->', translate('black'))
print('up ->', translate('up'))
