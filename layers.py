import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def Attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class Embedding(nn.Module):
    r"""Embedding layer constructs a vector vocabulary.

    Parameters:
        vocab_size: Indicate that how many word in vocabulary
        d_model: Indicate that how many dimensions a word has
    Example:
        embedding = nn.Embedding(10, 3) # an Embedding module containing ten word of 3 dimension
        input = torch.LongTensor([[1,2,4,5],[4,3,2,9]]) # a batch of 2 samples of 4 indices each
        e = embedding(input)
        print(e)
        >> tensor([[[ 1.4747, -1.1558,  1.3940],
                    [-0.3931,  1.0315, -0.1700],
                    [ 0.0427, -0.3308,  0.7080],
                    [ 0.8967,  0.2389,  1.6034]],

                   [[ 0.0427, -0.3308,  0.7080],
                    [ 1.3078, -0.4615,  0.9036],
                    [-0.3931,  1.0315, -0.1700],
                    [-0.0635,  2.6708,  0.8157]]], grad_fn=<EmbeddingBackward0>)
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 device: None,
                 dtype: None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.d_model = d_model

    def forward(self, x):
        r"""
        The reason why we need to multiply the square of the dimension
        is to prevent the data from being too small and having no effect
        after the subsequent Positional encoding superposition.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    r"""PositionalEncoding is to encode each word

    Parameters:
        len_position:
        d_model:
        dropout: parameter p of nn.dropout() that indicates probability of an element to be zeroed.Default: 0.5

    """
    def __init__(self,
                 len_position: int,
                 d_model: int,
                 dropout: float = 0.5,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.PE = torch.zeros(len_position, d_model)

        # position: torch.size(len_position, 1);
        position = torch.arange(len_position).unsqueeze(1)
        # position * div_term: (len_position * 1) * (1 * d_model) = (len_position * d_model);
        div_term = torch.exp(torch.arange(0, d_model, step=2) * (-(math.log(10000) / d_model)))
        self.PE[:, 0::2] = torch.sin(position * div_term)
        self.PE[:, 1::2] = torch.cos(position * div_term)

        # insert batch dimension by broadcasting, PE size:[1, len_position, d_model];
        self.PE = self.PE.unsqueeze(0)

    def forward(self, x):
        x += self.PE[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 head_num: int,
                 d_model: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model // head_num
        self.head_num = head_num
        self.dropout = dropout
        self.Linears = clones(nn.Linear(d_model, d_model), 4)
        self.Attention = None
        self.Dropout = nn.Dropout(p=dropout)

    def forward(self,
                query,
                key,
                value,
                mask=None,
                ):
        nbatch = query.size(0)

        seq_q_len = query.size(1)
        seq_k_len = key.size(1)
        seq_v_len = value.size(1)

        if mask is not None:
            mask = mask.unsqueeze(1)

        # Linear transforming
        query = self.Linears[0](query)
        key = self.Linears[1](key)
        value = self.Linears[2](value)

        query = query.view(nbatch, seq_q_len, self.head_num, self.d_k)
        key = key.view(nbatch, seq_k_len, self.head_num, self.d_k)
        value = value.view(nbatch, seq_v_len, self.head_num, self.d_k)

        # torch.size(nbatch, seq_q_len, head_num, d_k) -> torch.size(nbatch, head_num, seq_q_len, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        x, self.attn = Attention(query, key, value, mask=mask, dropout=self.Dropout)

        x = x.transpose(1, 2).contiguous()
        x = x.view(nbatch, -1, self.head_num * self.d_k)
        return self.Linear[-1](x)


class LayerNorm(nn.Module):
    r"""Used for Layer Normalization

    self.a_2: The trainable parameter self.a_2 magnifies or reduces the normalized data
    self.b_2: The trainable parameter self.b_2 translates the normalized data

    The purpose of the scaling and translation transformation is to restore the representation ability of the data. Because the direct normalization of the data (zero mean and unit variance standardization) accelerates the convergence of the model, it also destroys the original distribution of the data, which may lead to the loss of information.

    By introducing trainable scaling and translation parameters, the model can automatically learn how to linearly transform the normalized data to make its distribution close to the original data distribution again, thus retaining the representation power of the input data to the greatest extent.
    """
    def __init__(self,
                 features: int,
                 eps=1e-6,
                 ):
        super().__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # Suppose dimension of x is (nbatch, seq_length, 512);
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + self.eps

        return self.a_2 * (x - mean) / std + self.b_2


class PositionWiseFeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 dropout: float=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class SubLayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.LayerNorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.LayerNorm(x)))
    