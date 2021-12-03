import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len=5000):
        """
        初始化函数
        :param d_model:表示词嵌入的维度
        :param dropout:置0比例,介于[0, 1)之间的比例
        :param max_len:每个句子最大的长度
        """
        super(PositionalEncoding, self).__init__()

        # 实例化预定义的Dropout层，并将Dropout传入其中，获得对象的self.dropout
        self.dropout = nn.Dropout(dropout)

#         初始化一个位置编码矩阵，是一个0矩阵，大小是max_len * d_model
        pe = torch.zeros(max_len, d_model)

#         初始化一个绝对位置矩阵，在我们这里，词汇的绝对位置就是用它的索引去表示
#         所以我们首先使用arange方法获取一个连续自然数乡里那个，然后在使用unsequeeze方法拓展向量维度使其成为矩阵
#         又因为参数传的是1,代表矩阵拓展的位置，会是一个乡里那个变成max_len x l的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

#         绝对位置矩阵初始化之后，接下来就是考虑如何见这些位置信息加入到位置编码矩阵中，
#         最贱的思路就是先将max_len x l的绝对位置矩阵，变成max_lenxd_model形状，然后覆盖原来的初始位置编码矩阵即可，
#         要做这种矩阵变化，就需要一个lxd_model形状的变换矩阵div_term，我们对这个变换矩阵要求除了形状外，
#         有助于在之后的梯度下降过程中更快的收敛，这样我们就可以开始初始化这一个自然矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        # 偶数位置就进行正弦变换
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数列都用余弦波进给它赋值
        pe[:, 1::2] = torch.cos(position * div_term)

        # 这样就得到了位置编码矩阵pe,pe现在还只是一个二维矩阵，要想和embedding的输出(一个三维张量)相加
        # 拓展一个维度，所以这里使用unsqueeze拓展维度
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer,什么是buffer呢
        # 我们把它认为是对模型效果有帮助，但是却不知道是模型结构中超参数或者参数，不需要随着优化步骤而优化步骤进行更新的增益对象
#         注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        forward函数的参数x,表示文本序列的词嵌入表示
        :return:
        """
        # 在相加之前我们对pe做一些适配工作，将这三维张量的第二维也就是句子最大长度的那一维切
        # 因为我们默认max_len为5000一般来讲实在太大了，很难有一条句子包含5000个词汇，所以要及逆行
        # 最后使用Variable进行封装,使其与x的央视相同,但是它不需要进行梯度,因为不会随着梯度优化器进行迭代,因此把required_grad设置成False
        print(x.shape)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 最后使用self.dropout对象进行丢弃操作,并返回结果
        return self.dropout(x)


# 创建一张15 x 5 大小的画布
plt.figure(figsize=(15, 5))

# 实例化PositionalEncoding类得到PositionalEncoding对象
# ，输入参数是20和0默认max_len是5000
pe = PositionalEncoding(20, 0)

# 向pe传入Variable封装的tensor,这样pe会直接执行forward函数，
# 且这个tensor里的数值都是0，被处理后相当于编码张量
y = pe(Variable(torch.zeros(1, 100, 20)))

# 定义画布的横纵坐标，横坐标到100的长度，纵坐标是某一个词汇中的某个维度特征在不同长度下对应值
# 因为总共20维之多，所以我们这里只查看4， 5, 6,7组的值
plt.plot(np.arange(100), y[0, :, 4:8])

# 在画布上填写维度信息
plt.legend(['dim %d' %p for p in np.arange(4, 8)])
plt.show()