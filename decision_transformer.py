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
    def __init__(self, feature_dim, num_decoder_layer=4, d_model=256, d_ff=512, num_heads=8, dropout=0.1):
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

    def forward(self, x, pred_len = 1):
        if x.dim() == 2:
            x = x.unsqueeze(0) # add batch dimension if there is none

        batch_size, seq_len, feature_dim = x.size()
        blank = -2 * torch.ones(batch_size, pred_len, feature_dim).to(x.device) # the non blank elements are all between -1 and 1, hence can use -2 to represent blank
        x = torch.cat((x, blank), dim=1) # padding at the end of each trajectory
        seq_len += pred_len
        
        mask = self.causal_mask(seq_len).to(x.device)
        output = self.decoder(self.embed(x), mask)[:,-pred_len:,:]
        prediction = self.predictor(output)

        return prediction # (batch_size, pred_len, feature_dim)
    

class DecisionTransformer(torch.nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=64, max_traj_len=50, num_decoder_layer=4, d_model=256, d_ff=512, num_heads=8, dropout=0.1):
        super(DecisionTransformer, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.max_traj_len = max_traj_len # maximum length of the input trajectory

        self.obs_embed = torch.nn.Linear(state_dim, feature_dim)
        self.act_embed = torch.nn.Linear(action_dim, feature_dim)
        self.rtg_embed = torch.nn.Linear(1, feature_dim)

        self.act_predict = torch.nn.Linear(feature_dim, action_dim)

        self.transformer = Transformer(
            feature_dim, num_decoder_layer, d_model, d_ff, num_heads, dropout
        )

    def forward(self, rtg, state, memory):
        squeeze = False
        if x.dim() == 2:
            # add batch dimension if there is none
            x = x.unsqueeze(0) 
            rtg = rtg.unsqueeze(0)
            state = state.unsqueeze(0)
            squeeze = True
        rtg_encoding = F.tanh(self.rtg_embed(rtg)).unsqueeze(-2) # (batch_size, 1, feature_dim)
        obs_encoding = F.tanh(self.obs_embed(state)).unsqueeze(-2) # (batch_size, 1, feature_dim)
        x = torch.cat((memory, rtg_encoding, obs_encoding), dim=-2) # (batch_size, max_traj_len + 2, feature_dim)
        x = x[:, -self.max_traj_len:, :] # only use the last max_traj_len elements of the memory

        output = self.transformer(x, pred_len = 1) 
        action = self.act_predict(output) # predict the action, (batch_size, 1, action_dim)
        new_memory = torch.cat((x, output), dim=-2) # append the action encoding to the memory, (batch_size, max_traj_len, feature_dim)
        new_memory = new_memory[:, -self.max_traj_len:, :] # only use the last max_traj_len elements of the memory
        if squeeze:
            action = action.squeeze(0)
            new_memory = new_memory.squeeze(0)
        return action, new_memory