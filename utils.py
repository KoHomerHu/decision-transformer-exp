import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
import math
import pickle
import numpy as np


def stack_modules(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


"""query, key, value are of shape (batch_size, num_heads, seq_len, d_model)"""
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5) # (batch_size, num_heads, seq_len, seq_len)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.nn.functional.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.k = torch.nn.Parameter(torch.ones(features))
        self.b = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # mean of features
        std = x.std(-1, keepdim=True) # std of features
        return self.k * (x - mean) / (std + self.eps) + self.b
    

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = stack_modules(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)

        query, key, value = [
            l(x).view(num_batches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.num_heads * self.d_k)

        del query, key, value

        return self.linears[-1](x)
    

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

class ANN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANN, self).__init__()
        layer_dim = [input_dim,] + hidden_dim + [output_dim,]
        self.fc = torch.nn.ParameterList(
            [torch.nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)]
        )

    def forward(self, x):
        for layer in self.fc:
            x = F.relu(layer(x))
        return x
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    

"""Function to compute the reward-to-go"""
def reward_to_go(rewards):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class TrajectoryDataset(Dataset):
    def __init__(self, action_dim, pkl_file, root_dir = "./data/", max_traj_len = 100):
        self.action_dim = action_dim
        self.data = pickle.load(open(root_dir + pkl_file, 'rb'))
        self.data = list(filter(lambda x : len(x['state']) >= max_traj_len, self.data))
        self.max_traj_len = max_traj_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        trajectory = self.data[idx]
        N = len(trajectory['state'])
        start_idx = torch.randint(0, N - self.max_traj_len, (1,)).item()
        end_idx = start_idx + self.max_traj_len
        state = trajectory['state'][start_idx:end_idx]
        rtg = trajectory['reward-to-go'][start_idx:end_idx]
        action = trajectory['action'][start_idx:end_idx]
        return {
            'state' : torch.tensor(state),
            'rtg' : torch.tensor(rtg).unsqueeze(-1),
            'action' : F.one_hot(torch.tensor(action), self.action_dim)
        }
    

"""Helps to sample from the trajectory dataset multiple times"""
def cycle(iterable):
    while True:
        for x in iterable:
            yield x