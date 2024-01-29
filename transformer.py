import torch
import copy
from utils import *
    

class EncoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
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


class Encoder(torch.nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = stack_modules(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)    
    

class DecoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm1 = LayerNorm(size)

        self.src_attn = src_attn
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm2 = LayerNorm(size)

        self.feed_forward = feed_forward
        self.dropout3 = torch.nn.Dropout(dropout)
        self.norm3 = LayerNorm(size)

        self.size = size
        
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.src_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x
    

class Decoder(torch.nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = stack_modules(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # src -> embedding
        self.tgt_embed = tgt_embed # tgt -> embedding

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

class Transformer(torch.nn.Module):
    def __init__(self, src_entry, tgt_entry, N=3, d_model=128, d_ff=256, num_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, d_model)
        ff = FeedForward(d_model, d_ff)
        position = PositionalEncoding(d_model, dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            torch.nn.Sequential(Embeddings(d_model, src_entry), c(position)),
            torch.nn.Sequential(Embeddings(d_model, tgt_entry), c(position)),
            Generator(d_model, tgt_entry)
        )
        self.d_model = d_model

        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        
        def forward(self):
            pass