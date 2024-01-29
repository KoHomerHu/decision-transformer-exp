import torch
import copy
from utils import *
    

"""Modified implementation of decoder that does not rely on outputs from the encoder."""
class DecoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm1 = LayerNorm(size)

        self.feed_forward = feed_forward
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm2 = LayerNorm(size)

        self.size = size
        
    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x
    

class Decoder(torch.nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = stack_modules(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


"""
A decoder-only transformer model that has been modified to accept input as a sequence or a batch of 
sequences of 1D tensors (i.e. (batch_size, seq_len, state_dim) or (seq_len, state_dim)), and output 
the prediction of the next state only. Assume that the input is not all negative.
"""
class Transformer(torch.nn.Module):
    def __init__(self, feature_dim, N=4, d_model=256, d_ff=512, num_heads=8, dropout=0.1):
        super(Transformer, self).__init__()

        self.decoder=  Decoder(
            DecoderLayer(
                d_model, 
                MultiHeadedAttention(num_heads, d_model),
                FeedForward(d_model, d_ff), 
                dropout
            ), 
            N
        )

        self.embed = torch.nn.Sequential(
            ANN(feature_dim, [d_model * 2, d_model * 2], d_model), 
            PositionalEncoding(d_model, dropout)
        )

        self.predictor = ANN(d_model, [d_model * 2, d_model * 2], feature_dim)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def causal_mask(self, seq_len):
        attn_shape = (1, 1, seq_len, seq_len)
        causal_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return causal_mask == 0

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0) # add batch dimension if there is none

        batch_size, seq_len, feature_dim = x.size()
        blank = -torch.ones(batch_size, 1, feature_dim).to(x.device)
        x = torch.cat((x, blank), dim=1) # padding at the end of each trajectory
        seq_len += 1
        
        mask = self.causal_mask(seq_len).to(x.device)
        output = self.decoder(self.embed(x), mask)[:,-1,:]
        prediction = self.predictor(output)

        return prediction


class DecisionTransformer(torch.nn.Module):
    def __init__(self, state_dim, action_dim, K=50, N=4, d_model=256, d_ff=512, num_heads=8, dropout=0.1):
        super(DecisionTransformer, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K = K # length of the input trajectory

        self.transformer = Transformer(
            state_dim + action_dim + 1, N, d_model, d_ff, num_heads, dropout
        ) # the feature dimension is state_dim + action_dim + 1 (for reward-to-go)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0) # add batch dimension if there is none
        
        batch_size, seq_len, feature_dim = x.size()
        blank = -torch.ones(batch_size, 1, feature_dim).to(x.device)

        # padding at the beginning of each trajectory if length not enough
        if seq_len < self.K:
            x = torch.cat((blank.repeat(1, self.K - seq_len, 1), x), dim=1) 

        # get prediction of the next action from the transformer
        output = self.transformer(x) # shape of (batch_size, state_dim + action_dim + 1)
        _, action, _ = torch.split(output, [self.state_dim, self.action_dim, 1], dim=1) 

        return F.softmax(action, dim=1)
    