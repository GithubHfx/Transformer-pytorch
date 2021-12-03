# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from Attention import attention
#
# # 首先需要定义克隆函数，因为在多头注意力机制的视线中，
# # 用到多个结构相同的线性层
# # 我们将使用函数将他们一同初始化在一个网络层列表中，之后的结构中也会用到该函数
# def clones(module, N):
#     """
#     用于生成相同网络层的克隆函数，它的参数module表示要克隆的目标网络层，N代表要克隆的数量
#     :param module:
#     :param N:
#     :return:
#     """
#     # 在函数中，我们通过for循环对module进行N词深度拷贝，使得其中每个
#     # module成为独立的层
#     # 然后将其放在nn.ModuleList类型的列表中去存放
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, head, embedding_dim, dropout=0.1):
#         """
#         类初始化的时候传入3个参数
#         :param head:代表头数
#         :param embedding_dim:词嵌入维度
#         :param dropout:置零比例
#         """
#
#         super(MultiHeadAttention, self).__init__()
#
# #         在函数中首先使用一个测试中常用的assert语句，也就是判断h是否能够被d_model整除，
#         # 这是因为我们之后要给每个头分配等量的此特征也就是embedding_dim/head个
#         assert embedding_dim % head == 0
#
#         # 得到的没个头获得的分割词向量维度d_k
#         self.d_k = embedding_dim // head
#
#         # 传入头数h
#         self.head = head
#
#         # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim  x embedding_dim embedding_dim
#         # 为什么是四个呢，这是因为在多头注意力中，Q、K、V各需要一个，最后凭借的矩阵也需要一个
#         self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
#
#         # self.attn为None，代表最后的注意力张量，现在还没有结果所以为None
#         self.attn = None
#         # 最后就是一个dropout,它通过nn中的Dropout实例化而来，置零比例为传入超参数
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, query, key, value, mask=None):
#         """
#         前向逻辑函数，它的输入参数有四个前三个为
#         :param query: 注意力需要的全文信息
#         :param key: 提示信息
#         :param value: 最后的输出信息
#         :param mask: 掩码，在解码器中需要用到的
#         :return: 返回注意力计算结果
#         """
#         # 如果存在掩码张量mask
#         if mask is not None:
#             # 使用unsqueeeze拓展维度，代表多头中的第n头
#             mask = mask.unsqueeze(1)
#
#         # 接着，获得一个batch_size变量，他是query尺寸的第1个数字，代表有多少条样本
#         batch_size = query.size(0)
#
#         # 之后进入多头处理环节
#         # 首先利用zip将输入QKV与三个线性层组到一启，然后循环将QKV送入夏宁变换
#         # 昨晚线性变换后开始分割每个头的输入，使用view方法对线性变换结果进行维度的重塑，多增加一维
#         # 这样做意味着每个头可以获得一部分词特征组成的句子，
#         # 计算机这样做会根据这种变换自动计算这里的值然后对第二位和第三位进行转置操作
#         # 为了让代表句子长度的维度和词向量的维度相邻，这样注意力机制才能找到词义与句子位置之间的关系
#         query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
#                              for model, x in zip(self.linears,(query, key, value))]
#
#         # 得到每个头的输入后，接下来就是将他们传入到attention中，
#         # 这里直接调用我们之前实现的attention函数，同时也将mask和dropout传入其中
#         x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
#
#         # 通过多头注意力计算之后我们就得到了每个头就四环结果组成的4维张量
#         # 我们需要将其转换为输入形状，因此这里开始进行第一步处理的逆操作
#         # contiguous方法能够让转之后的向量使用view方法，否则无法直接使用
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
#
#         # 最后concat之后最后一个线性层进行线性变换得到最终多头注意力结构的输出结果
#         return self.linears[-1](x)
#
#
# # # tensor.view()演示
# # x = torch.randn(4, 4)
# # print(x, x.size())
# # y = x.view(16)
# # print(y, y.size())
# # z = x.view(-1, 8)
# # print(z, z.size())
# #
# # a = torch.randn(1, 2, 3, 4)
# # print(a, a.size())
# # b = a.transpose(1, 2) # 交换第二个维度和第三个维度，下标从0开始
# # print(b, b.size())
# # c = a.view(1, 3, 2, 4) #不改变在内存中的层结构
# # print(c, c.size())
# #
