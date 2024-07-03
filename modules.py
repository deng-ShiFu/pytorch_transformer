import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


# 多头注意力
class MultiHeadAttion(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttion, self).__init__()
        # 确保模型维度(d_model)可以被头数量(num_heads)整除
        assert d_model % num_heads == 0, "模型维度必须可以被头数量整除"
        
        # 初始化维度
        self.d_model = d_model #模型维度
        self.num_heads = num_heads #头数量
        self.d_k = d_model // num_heads #每个头的Key，Quert和Value的维度
        
        # 多头注意力中的线性层Linear Layers
        # 包含Q、K、V的三个权重，用于对输入进行线性变换产生Q、K、V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model) # 输出的转换

    
    # 计算注意力分数
    def scaled_dot_product_attion(self, Q, K, V, mask = None):
        # 计算注意力得分
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))/ math.sqrt(self.d_k)
        # 添加掩码，如果提供了掩码
        '''
        if (mask is not None) and (mask.nelement() != 0):
            # masked_fill(mask, value)
            # 对于mask中为True（或者非零）的位置，将attn_scores中对应位置的值替换为value。
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        '''
        # 使用softmax将得分转换为权重
        attn_probs = torch.softmax(attn_scores, dim = -1)
        # 使用注意力概论对Value矩阵进行加权得到输出
        output = torch.matmul(attn_probs, V)
        return output
    
    # 用于将x分割成多个头
    def split_heads(self, x):
        # 批大小， 序列长度(单词数量？)， (单词划分的长度？)
        batch_size, seq_length, d_model = x.size()
        # 返回 批大小，头数量，序列长度，每个头的维度
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        #return x.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    # 用于合并多个头
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        #return x.transpose(1, 2).reshape(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask = None):
        # 先使用线性变换产生Q，K，V。然后再进行分裂
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 计算多头注意力
        attn_output = self.scaled_dot_product_attion(Q, K, V, mask)

        # 合并多个头然后计算输出
        output = self.W_o(self.combine_heads(attn_output))
        return output


# 位置前馈网络(Position-wise Feed-Forward Networks)
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        # 线性层
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

# 位置编码(Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # max_seq_length宽，d_model长的0矩阵，用于存储位置编码
        pe = torch.zeros(max_seq_length, d_model)
        # 包含序列中每个位置的位置索引的张量，从0到max_seq_length(不包括)的一维张量，步长为1
        # unsqueeze在张量指定维度位置插入一个额外维度，形成一个二维的(max_seq_length,1)形状的矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # 选择所有的行，从第0列开始，每隔一列选择一列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
            

    def forward(self, x):
        # 将位置编码添加到输入 x。选择pe所有的行，从第0列到x.size(1)(seq_length)的列
        return x + self.pe[:, :x.size(1)]


# 编码器
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttion(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        # 残差连接
        x  = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

# 解码器
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttion(d_model, num_heads)
        self.cross_attn = MultiHeadAttion(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
        
