# 对于一些简单的工程应用和一些简单的研究research都够了
# from transformers import BertModel
# model = BertModel.from_pretrained("bert-base-uncased")
# 1.Transformer回归
# 2.用Pytorch实现Transformer
# 3.将自实现与官方实现比较有什么差别
# 输入是N * d的张量x，其中N代表序列的长度，d代表每个词的维度
# 将x输入到Multi-head self-attention 输出与输出形状是一样的也是n * d的张量，
# 再跟input做一个相加，做一个ResidualConnection，将输出在经过一个LayerNorm
# 经过Feedforward Network再次与输入进行ResidualConnection，经过LayerNorm输出N * d张量，一层Transformer架构完成了
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


# 构建Embedding类实现文本嵌入

class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        """
        :param d_model:词嵌入输出的维度
        :param vocab: 此表的大小
        :return:
        """
        super(Embedding, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # 将参数传入类中
        self.d_model = d_model

    def forward(self, x):
        """

        :param x:代表输入进模型的文本通过词汇映射后的数字张量
        :return:
        """
        return self.lut(x) * math.sqrt(self.d_model)


# Dropout演示:最后的矩阵会有4个零
# m = nn.Dropout(p=0.2)
# input1 = torch.randn(4, 5)
# output = m(input1)
# print(output)
# x = torch.tensor([1, 2, 3, 4])
# # 压缩成一行:1 * n的向量
# print(torch.unsqueeze(x, 0))
#
# # 压缩成一列n * 1的向量
# print(torch.unsqueeze(x, 1))
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
        # print(x.shape)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 最后使用self.dropout对象进行丢弃操作,并返回结果
        return self.dropout(x)


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


# 首先需要定义克隆函数，因为在多头注意力机制的视线中，
# 用到多个结构相同的线性层
# 我们将使用函数将他们一同初始化在一个网络层列表中，之后的结构中也会用到该函数
def clones(module, N):
    """
    用于生成相同网络层的克隆函数，它的参数module表示要克隆的目标网络层，N代表要克隆的数量
    :param module:
    :param N:
    :return:
    """
    # 在函数中，我们通过for循环对module进行N词深度拷贝，使得其中每个
    # module成为独立的层
    # 然后将其放在nn.ModuleList类型的列表中去存放
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        类初始化的时候传入3个参数
        :param head:代表头数
        :param embedding_dim:词嵌入维度
        :param dropout:置零比例
        """

        super(MultiHeadAttention, self).__init__()

        #         在函数中首先使用一个测试中常用的assert语句，也就是判断h是否能够被d_model整除，
        # 这是因为我们之后要给每个头分配等量的此特征也就是embedding_dim/head个
        assert embedding_dim % head == 0

        # 得到的没个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        # 传入头数h
        self.head = head

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim  x embedding_dim embedding_dim
        # 为什么是四个呢，这是因为在多头注意力中，Q、K、V各需要一个，最后凭借的矩阵也需要一个
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn为None，代表最后的注意力张量，现在还没有结果所以为None
        self.attn = None
        # 最后就是一个dropout,它通过nn中的Dropout实例化而来，置零比例为传入超参数
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向逻辑函数，它的输入参数有四个前三个为
        :param query: 注意力需要的全文信息
        :param key: 提示信息
        :param value: 最后的输出信息
        :param mask: 掩码，在解码器中需要用到的
        :return: 返回注意力计算结果
        """
        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeeze拓展维度，代表多头中的第n头
            mask = mask.unsqueeze(0)

        # 接着，获得一个batch_size变量，他是query尺寸的第1个数字，代表有多少条样本
        batch_size = query.size(0)

        # 之后进入多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一启，然后循环将QKV送入夏宁变换
        # 昨晚线性变换后开始分割每个头的输入，使用view方法对线性变换结果进行维度的重塑，多增加一维
        # 这样做意味着每个头可以获得一部分词特征组成的句子，
        # 计算机这样做会根据这种变换自动计算这里的值然后对第二位和第三位进行转置操作
        # 为了让代表句子长度的维度和词向量的维度相邻，这样注意力机制才能找到词义与句子位置之间的关系
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]

        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数，同时也将mask和dropout传入其中
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算之后我们就得到了每个头就四环结果组成的4维张量
        # 我们需要将其转换为输入形状，因此这里开始进行第一步处理的逆操作
        # contiguous方法能够让转之后的向量使用view方法，否则无法直接使用
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后concat之后最后一个线性层进行线性变换得到最终多头注意力结构的输出结果
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化函数三个参数
        :param d_model:词嵌入的输入维度
        :param d_ff:因为我们希望输入通过前馈全连接层后输入和输出的维度不变. 第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出维度.
        :param dropout:dropout置零比率
        """
        super(PositionwiseFeedForward, self).__init__()
        # 首先按照我们预期使用nn实例化了两个线性层对象，self.w1和self.w2
        # 他们的参数分别是d_model, d_ff和d_ff,  d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        # 然后使用nn实例化对象的self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播函数,
        :param x: 代表来自上一层的输出
        :return:
        """
        # 首先经过一个线性层,然后使用Functinal中的relu函数进行激活
        # 之后在使用dropout进行随机置0,最后通过第二个线性层w2,返回最终结果
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        输入两个参数
        :param features:features是特征维度
        :param eps: 很小的数防止除0情况出现
        """

        super(LayerNorm, self).__init__()

        # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数，
        # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，
        # 使其即能满足规范化要求，又能不改变针对目标的表征.最后使用nn.parameter封装，代表他们是模型的参数。

        # 初始化两个参数张量，用于对结果进行规范化操作
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        """
        进行规范化, 对x最后一个维度求均值操作
        :param x:
        :return:
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a * (x - mean) / (self.eps + std) + self.b


d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

emb = Embedding(d_model, vocab)
embr = emb(x)
# print("embr:", embr)
# print(embr.shape)
dropout = 0.1

# 句子最大长度
max_len = 60
# 将词嵌入作为位置编码器的输入，进行编码
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(embr)
print("pe_result:", pe_result)

# 令位置编码的输出
query, key, value = [pe_result] * 3

attn_scores, attn_map = attention(query, key, value)
print("attn:", attn_scores)
print("attn_map:", attn_map)

n_head = 8
embedding_dimension = 512
dropout_rate = 0.2
mask = Variable(torch.zeros(8, 4, 4))
mha = MultiHeadAttention(n_head, embedding_dimension, dropout_rate)
mha_result = mha(query, key, value, mask)
print("multi-head attention result", mha_result)
print("multi-head attention result shape", mha_result.shape)

d_ff_ = 64  # （128也可以)
pff = PositionwiseFeedForward(d_model, d_ff_, 0.2)
ff_result = pff(mha_result)
print("position wise feedforward network result", ff_result)
print("position wise feedforward network result shape", ff_result.shape)

features = 512
eps = 1e-6
layer_norm = LayerNorm(features, eps)
layer_norm_result = layer_norm(ff_result)
print("norm result: ", layer_norm_result)
print("norm result shape: ", layer_norm_result.shape)


class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        """
        子层连接对象初始化函数
        :param dim: 特征维度
        :param dropout_rate: 置零比率
        """
        super(SublayerConnection, self).__init__()

        # 实例化一个规范化层对象
        self.norm = LayerNorm(dim, 1e-6)

        # 数理化一个dropout对象
        self.dropout = nn.Dropout(dropout_rate)

        self.size = dim

    def forward(self, x, sublayer):
        """
        前馈函数
        :param x: 上一层输出结果用于进行规范化并进行残差链接
        :param sublayer: 子层
        :return: 返回连接结果
        """

        return x + self.dropout(sublayer(self.norm(x)))


# sc = SublayerConnection(embedding_dimension, dropout_rate)
# self_attention = MultiHeadAttention(n_head, embedding_dimension,dropout_rate)
# sublayer = lambda x: self_attention(x, x, x, torch.zeros(8, 4, 4))
# sc_result = sc(pe_result, sublayer)
# print(sc_result)
# print(sc_result.shape)


# class Encoder(nn.Module):
#     def __init__(self, n_head, dim, vocab, sentence_max_len, mask=torch.zeros(8, 4, 4), dropout_rate=0.1):
#         """
#
#         :param n_head:注意力头数
#         :param dim:特征维数
#         :param vocab:词汇个数
#         :param sentence_max_len: 句子最大长度
#         :param mask: 掩码张量
#         :param dropout_rate:置零比率
#         """
#         super(Encoder, self).__init__()
#
#         self.embedding = Embedding(dim, vocab)
#
#         self.pe = PositionalEncoding(dim, dropout_rate, sentence_max_len)
#
#         self.mha = MultiHeadAttention(n_head, dim, dropout_rate)
#
#         self.norm = LayerNorm(dim)
#
#         self.pff = PositionwiseFeedForward(dim, 128)
#
#         self.sublayer_connection = SublayerConnection(dim, dropout_rate)
#
#         self.mask = mask
#
#     def forward(self, x):
#         """
#
#         :param x:
#         :return:
#         """
#         embedding_result = self.embedding(x)
#
#         pe_result = self.pe(embedding_result)
#
#         sublayer1 = lambda x_: self.mha(x_, x_, x_, self.mask)
#
#         sc_result1 = self.sublayer_connection(pe_result, sublayer1)
#
#         sublayer2 = lambda x_: self.pff(x_)
#
#         sc_result2 = self.sublayer_connection(sc_result1, sublayer2)
#
#         return sc_result2
#
# n_head = 8
# embedding_dimension = 512
# vocab = 1000
# max_len = 60
# dropout_rate = 0.2
#
# encoder = Encoder(n_head, embedding_dimension, vocab, max_len, dropout_rate=dropout_rate)
# x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# encoding_result = encoder(x)
# print(encoding_result)
# print(encoding_result.shape)


# 使用EncoderLayer类实现编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """它的初始化函数参数有四个，分别是size，其实就是我们词嵌入维度的大小，它也将作为我们编码器层的大小,
           第二个self_attn，之后我们将传入多头自注意力子层实例化对象, 并且是自注意力机制,
           第三个是feed_froward, 之后我们将传入前馈全连接层实例化对象, 最后一个是置0比率dropout."""
        super(EncoderLayer, self).__init__()

        # 首先将self_attn和feed_forward传入其中.
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 把size传入其中
        self.size = size

    def forward(self, x, mask):
        """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask."""
        # 里面就是按照结构图左侧的流程. 首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层. 最后返回结果.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


head = 8
d_model = 512
d_ff = 64
x = pe_result
dropout = 0.2
self_attn = MultiHeadAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(8, 4, 4))
encoder_layer = EncoderLayer(d_model, self_attn, ff, dropout)
print("Encoder layer result: ", encoder_layer(pe_result, mask))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        传入encoder_layer进行
        :param layer: 传入encoder层对象
        :param N: 需要几层堆叠
        """
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)

        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        forward函数的输入和编码器层相同
        :param x:上一层的输出
        :param mask:代表掩码张量
        :return:
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


size = 512
head = 8
d_model = 512
d_ff = 64
c = copy.deepcopy
attn = MultiHeadAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)

# 编码器中编码器层的个数N
N = 8
mask = Variable(torch.zeros(8, 4, 4))

en = Encoder(layer, N)
en_result = en(x, mask)
print(en_result)
print(en_result.shape)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attention, cross_attention, feed_forward, dropout=0.1):
        """
        解码器层
        :param size: 特征维度
        :param self_attention: 自注意力对象
        :param cross_attention: 交叉注意力对象
        :param feed_forward: 前馈神经网络
        :param dropout: 置零比率
        """
        super(DecoderLayer, self).__init__()

        self.size = size

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.dropout = dropout

        self.sublayers = clones(SublayerConnection(self.size, dropout_rate=self.dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """

        :param x:目标的embedding张量
        :param memory:编码器的输出张量
        :param source_mask: 编码掩码，
        :param target_mask:目标掩码
        :return:
        """
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, target_mask))

        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x; k，v是编码层输出memory，
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄漏，而是遮蔽掉对结果没有意义的字符而产生的注意力值，
        # 以此提升模型效果和训练速度. 这样就完成了第二个子层的处理.
        x = self.sublayers[1](x, lambda x: self.cross_attention(x, m, m, source_mask))

        return self.sublayers[2](x, self.feed_forward)


head = 8
size = 512
d_model = 512
d_ff = 64
dropout = 0.2
self_attn = src_attn = MultiHeadAttention(head, d_model, dropout)

# 前馈全连接层也和之前相同
ff = PositionwiseFeedForward(d_model, d_ff, dropout)

# x是来自目标数据的词嵌入表示, 但形式和源数据的词嵌入表示相同, 这里使用per充当.
x = pe_result

# memory是来自编码器的输出
memory = en_result

# 实际中source_mask和target_mask并不相同, 这里为了方便计算使他们都为mask
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
print(dl_result)
print(dl_result.shape)


# 使用类Decoder来实现解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        """初始化函数的参数有两个，第一个就是解码器层layer，第二个是解码器层的个数N."""
        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层.
        # 因为数据走过了所有的解码器层后最后要做规范化处理.
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        """forward函数中的参数有4个，x代表目标数据的嵌入表示，memory是编码器层的输出，
           source_mask, target_mask代表源数据和目标数据的掩码张量"""

        # 然后就是对每个层进行循环，当然这个循环就是变量x通过每一个层的处理，
        # 得出最后的结果，再进行一次规范化返回即可.
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# 分别是解码器层layer和解码器层的个数N
size = 512
d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = MultiHeadAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
N = 8

# 输入参数与解码器层的输入参数相同
x = pe_result
memory = en_result
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

# 调用
de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)
print(de_result)
print(de_result.shape)


# 将线性层和softmax计算层一起实现, 因为二者的共同目标是生成最后的结构
# 因此把类的名字叫做Generator, 生成器类
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """初始化函数的输入参数有两个, d_model代表词嵌入维度, vocab_size代表词表大小."""
        super(Generator, self).__init__()
        # 首先就是使用nn中的预定义线性层进行实例化, 得到一个对象self.project等待使用,
        # 这个线性层的参数有两个, 就是初始化函数传进来的两个参数: d_model, vocab_size
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """前向逻辑函数中输入是上一层的输出张量x"""
        # 在函数中, 首先使用上一步得到的self.project对x进行线性变化,
        # 然后使用F中已经实现的log_softmax进行的softmax处理.
        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复.
        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数,
        # 因此对最终我们取最大的概率值没有影响. 最后返回结果即可.
        return F.log_softmax(self.project(x), dim=-1)


# 词嵌入维度是512维
d_model = 512

# 词表大小是1000
vocab_size = 1000

# 输入x是上一层网络的输出, 我们使用来自解码器层的输出
x = de_result
gen = Generator(d_model, vocab_size)
gen_result = gen(x)
print(gen_result)
print(gen_result.shape)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embedding, target_embedding, generator):
        """
        编码器解码器架构
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param source_embedding: 源数据embedding对象
        :param target_embedding: 目标数据embedding对象
        :param generator: 生成器
        """
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embedding
        self.target_embed = target_embedding
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        编码器解码器前馈函数
        :param source: 源
        :param target: 目标
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        :return:

        """
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """
        解码器
        :param memory: 编码器中的输出
        :param source_mask:
        :param target:
        :param target_mask:
        :return:
        """
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6,
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    """该函数用来构建模型, 有7个参数，分别是源数据特征(词汇)总数，目标数据特征(词汇)总数，
       编码器和解码器堆叠数，词向量映射维度，前馈全连接网络中变换矩阵的维度，
       多头注意力结构中的多头数，以及置零比率dropout."""

    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，
    # 来保证他们彼此之间相互独立，不受干扰.
    c = copy.deepcopy

    # 实例化了多头注意力类，得到对象attn
    attn = MultiHeadAttention(head, d_model)

    # 然后实例化前馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model, dropout)

    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层.
    # 在编码器层中有attention子层以及前馈全连接子层，
    # 在解码器层中有两个attention子层以及前馈全连接层.
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, source_vocab), c(position)),
        nn.Sequential(Embedding(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embedding = nn.Embedding(vocab_size, d_model)
target_embedding = nn.Embedding(vocab_size, d_model)
generator = gen
source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
source_mask = target_mask = Variable(torch.zeros(8, 4, 4))
ed = EncoderDecoder(encoder, decoder, source_embedding, target_embedding, generator=generator)
ed_result = ed(source, target, source_mask, target_mask)
print(ed_result)
print(ed_result.shape)

# 导入工具包Batch, 它能够对原始样本数据生成对应批次的掩码张量
from pyitcast.transformer_utils import Batch, greedy_decode, get_std_opt
# 导入模型单轮训练工具包run_epoch, 该工具将对模型使用给定的损失函数计算方法进行单轮参数更新.
# 并打印每轮参数更新的损失结果.
from pyitcast.transformer_utils import run_epoch

# 导入标签平滑工具包, 该工具用于标签平滑, 标签平滑的作用就是小幅度的改变原有标签值的值域
# 因为在理论上即使是人工的标注数据也可能并非完全正确, 会受到一些外界因素的影响而产生一些微小的偏差
# 因此使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合. 通过下面示例了解更多.
from pyitcast.transformer_utils import LabelSmoothing


# 导入损失计算工具包, 该工具能够使用标签平滑后的结果进行损失的计算,
# 损失的计算方法可以认为是交叉熵损失函数.
from pyitcast.transformer_utils import  SimpleLossCompute


def data_generator(V, batch, num_batch):
    """该函数用于随机生成copy任务的数据, 它的三个输入参数是V: 随机生成数字的最大值+1,
       batch: 每次输送给模型更新一次参数的数据量, num_batch: 一共输送num_batch次完成一轮
    """
    # 使用for循环遍历nbatches
    for i in range(num_batch):
        # 在循环中使用np的random.randint方法随机生成[1, V)的整数,
        # 分布在(batch, 10)形状的矩阵中, 然后再把numpy形式转换称torch中的tensor.
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data = torch.LongTensor(data.long())
        # print("data:", data)
        # 接着使数据矩阵中的第一列数字都为1, 这一列也就成为了起始标志列,
        # 当解码器进行第一次解码的时候, 会使用起始标志列作为输入.
        data[:, 0] = 1

        # 因为是copy任务, 所有source与target是完全相同的, 且数据样本作用变量不需要求梯度
        # 因此requires_grad设置为False
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
        yield Batch(source, target)


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()

        run_epoch(data_generator(V, 8, 20), model, loss)

        model.eval()

        run_epoch(data_generator(V, 8, 5), model, loss)

    # 模型进入测试模式
    model.eval()

    # 假定的输入张量
    source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]]))

    # 定义源数据掩码张量, 因为元素都是1, 在我们这里1代表不遮掩
    # 因此相当于对源数据没有任何遮掩.
    source_mask = Variable(torch.ones(1, 1, 10))

    # 最后将model, src, src_mask, 解码的最大长度限制max_len, 默认为10
    # 以及起始标志数字, 默认为1, 我们这里使用的也是1
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


# model和loss都是来自上一步的结果

if __name__ == '__main__':
    # 进行10轮训练
    epochs = 10

    # 将生成0-10的整数
    V = 11

    # 每次喂给模型20个数据进行参数更新
    batch = 20

    # 连续喂30次完成全部数据的遍历, 也就是1轮
    num_batch = 30
    # 使用make_model获得model
    model = make_model(V, V, N=2)

    # 使用get_std_opt获得模型优化器
    model_optimizer = get_std_opt(model)

    # 使用LabelSmoothing获得标签平滑对象
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

    # 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

    # 使用LabelSmoothing实例化一个crit对象.
    # 第一个参数size代表目标数据的词汇总数, 也是模型最后一层得到张量的最后一维大小
    # 这里是5说明目标词汇总数是5个. 第二个参数padding_idx表示要将那些tensor中的数字
    # 替换成0, 一般padding_idx=0表示不进行替换. 第三个参数smoothing, 表示标签的平滑程度
    # 如原来标签的表示值为1, 则平滑后它的值域变为[1-smoothing, 1+smoothing].
    crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)

    # 假定一个任意的模型最后输出预测结果和真实结果
    predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                          [0, 0.2, 0.7, 0.1, 0],
                                          [0, 0.2, 0.7, 0.1, 0]]))

    # 标签的表示值是0，1，2
    target = Variable(torch.LongTensor([2, 1, 0]))

    # 将predict, target传入到对象中
    crit(predict, target)

    # 绘制标签平滑图像
    plt.imshow(crit.true_dist)
    plt.show()
    run(model, loss)
