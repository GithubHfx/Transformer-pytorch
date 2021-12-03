import math

import torch
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    """
    注意力机制实现
    :param query:
    :param key:
    :param value:
    :param mask: 掩码张量
    :param dropout: nn.Dropout的实例化对象, 默认为None
    :return:
    """
    # 在函数中,首先取query的最后一维大小，一般情况下就等同于我们的词嵌入维度，
    # 命名为d_k
    d_k = query.size(-1)

    # 按照注意力机制，将query于key的转置相乘,这里的key是将最后两个维度进行转置，再除以缩放系数
    # 得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print(scores.size())
    # 判断是否使用掩码张量
    # 如果使用那么应该是在解码器使用，不适用就是在Encoder里面使用
    if mask is not None:
        # 使用tensor的masked_fill方法,将掩码张量和scores张量每个位置一一比较，如果掩码张量中为0的，
        # 则对应socres用-1e9这个值来替换,非常小，也就意味着这个位置的值不可能被选中
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对score的最后一维进行softmax操作，使用F.softmax方法，第一个参数是softmax对象，第二个参数
    # 这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim=-1)

#     之后判断是否使用dropout进行置0
    if dropout is not None:
        #  将p_attn传入dropout对象中进行舍弃操作
        p_attn = dropout(p_attn)

    # 返回softmax结果与value的值，也返回p_attn
    return torch.matmul(p_attn, value), p_attn
