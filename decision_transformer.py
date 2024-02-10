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
    def __init__(self, feature_dim, num_decoder_layer=6, d_model=256, d_ff=512, 
                 num_heads=8, dropout=0.1, positional_encoding = True):
        super(Transformer, self).__init__()

        self.decoder=  Decoder(
            DecoderLayer(
                d_model, 
                MultiHeadedAttention(num_heads, d_model),
                FeedForward(d_model, d_ff), 
                dropout
            ), 
            num_decoder_layer
        )

        if positional_encoding:
            self.embed = torch.nn.Sequential(
                ANN(feature_dim, [d_model * 2, d_model * 2], d_model), 
                PositionalEncoding(d_model, dropout)
            )
        else:
            self.embed = ANN(feature_dim, [d_model * 2, d_model * 2], d_model)

        self.predictor = ANN(d_model, [d_model * 2, d_model * 2], feature_dim)

        self.d_model = d_model

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def causal_mask(self, seq_len, diagonal = 1):
        attn_shape = (1, 1, seq_len, seq_len)
        causal_mask = torch.triu(torch.ones(attn_shape), diagonal=diagonal).type(torch.uint8)
        return causal_mask == 0

    def forward(self, x, pred_len = 1, padding = True, diagonal = 1):
        if x.dim() == 2:
            x = x.unsqueeze(0) # add batch dimension if there is none

        batch_size, seq_len, feature_dim = x.size()
        if padding:
            blank = -2 * torch.ones(batch_size, pred_len, feature_dim).to(x.device) # the non blank elements are all between -1 and 1, hence can use -2 to represent blank
            x = torch.cat((x, blank), dim=1) # padding at the end of each trajectory
            seq_len += pred_len
        
        mask = self.causal_mask(seq_len, diagonal).to(x.device)
        output = self.decoder(self.embed(x), mask)[:,-pred_len:,:]
        prediction = self.predictor(output)

        return prediction # (batch_size, pred_len, feature_dim)
    

class DecisionTransformer(torch.nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=64, num_decoder_layer=8, 
                 d_model=128, d_ff=512, num_heads=8, dropout=0.1):
        super(DecisionTransformer, self).__init__()

        self.d_model = d_model

        self.transformer = Transformer(
            feature_dim, num_decoder_layer, d_model, d_ff, num_heads, dropout, positional_encoding = False
        )

        self.rtg_embed = torch.nn.Linear(1, feature_dim)
        self.state_embed = torch.nn.Linear(state_dim, feature_dim)
        self.act_embed = torch.nn.Linear(action_dim, feature_dim)
        self.positional_encoding = PositionalEncoding(feature_dim, dropout)

        self.act_predict = torch.nn.Linear(feature_dim, action_dim)

    def forward(self, rtg, obs, act):
        rtg = F.tanh(self.rtg_embed(rtg))
        rtg = self.positional_encoding(rtg)
        batch_size, seq_len, feature_dim = rtg.shape
        obs = F.tanh(self.state_embed(obs))
        obs = self.positional_encoding(obs)
        act = F.tanh(self.act_embed(act))
        blank = -2 * torch.ones(act.shape[0], 1, act.shape[-1]).to(act.device)
        act = torch.cat((act, blank), dim=1)
        act = self.positional_encoding(act)
        input = torch.cat((rtg, obs, act), dim=-1)
        input = input.reshape(batch_size, seq_len * 3, feature_dim)
        output = self.transformer(input, pred_len = 1, padding = False)
        prob = F.softmax(self.act_predict(output), dim=-1).squeeze(1)
        return prob


"""
Alternative implementation of the decision transformer that does not embed the rtgs, obs and act into 
the same feature space before feeding them into the transformer. 
"""
class DecisionTransformer2(torch.nn.Module):
    def __init__(self, state_dim, action_dim, num_decoder_layer=12, 
                 d_model=256, d_ff=512, num_heads=8, dropout=0.1):
        super(DecisionTransformer2, self).__init__()

        self.action_dim = action_dim
        self.d_model = d_model

        self.transformer = Transformer(
            1 + state_dim + action_dim, num_decoder_layer, d_model, d_ff, num_heads, dropout
        )

    def forward(self, rtg, obs, act):
        blank = -2 * torch.ones(act.shape[0], 1, act.shape[-1]).to(act.device)
        act = torch.cat((act, blank), dim=1)
        input = torch.cat((rtg, obs, act), dim=-1)
        output = self.transformer(input, pred_len = 1, padding = False)
        output = output.squeeze(1)
        prob = F.softmax(output[:,-self.action_dim:], dim=-1)
        return prob
