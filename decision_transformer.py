import torch
import copy
from utils import *
    

# class EncoderLayer(torch.nn.Module):
#     def __init__(self, size, self_attn, feed_forward, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.dropout1 = torch.nn.Dropout(dropout)
#         self.norm1 = LayerNorm(size)

#         self.feed_forward = feed_forward
#         self.dropout2 = torch.nn.Dropout(dropout)
#         self.norm2 = LayerNorm(size)

#         self.size = size
        
#     def forward(self, x, mask):
#         x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
#         x = self.norm2(x + self.dropout2(self.feed_forward(x)))
#         return x


# class Encoder(torch.nn.Module):
#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = stack_modules(layer, N)
#         self.norm = LayerNorm(layer.size)
        
#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)    
    

# class DecoderLayer(torch.nn.Module):
#     def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#         super(DecoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.dropout1 = torch.nn.Dropout(dropout)
#         self.norm1 = LayerNorm(size)

#         self.src_attn = src_attn
#         self.dropout2 = torch.nn.Dropout(dropout)
#         self.norm2 = LayerNorm(size)

#         self.feed_forward = feed_forward
#         self.dropout3 = torch.nn.Dropout(dropout)
#         self.norm3 = LayerNorm(size)

#         self.size = size
        
#     def forward(self, x, memory, src_mask, tgt_mask):
#         x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
#         x = self.norm2(x + self.dropout2(self.src_attn(x, memory, memory, src_mask)))
#         x = self.norm3(x + self.dropout3(self.feed_forward(x)))
#         return x
    

# class Decoder(torch.nn.Module):
#     def __init__(self, layer, N):
#         super(Decoder, self).__init__()
#         self.layers = stack_modules(layer, N)
#         self.norm = LayerNorm(layer.size)
        
#     def forward(self, x, memory, src_mask, tgt_mask):
#         for layer in self.layers:
#             x = layer(x, memory, src_mask, tgt_mask)
#         return self.norm(x)


# class EncoderDecoder(torch.nn.Module):
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator 

#     def forward(self, src, tgt, src_mask, tgt_mask):
#         return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)
    
#     def decode(self, memory, src_mask, tgt, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

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
        
    def forward(self, x, tgt_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x
    

class Decoder(torch.nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = stack_modules(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


"""
A decoder-only transformer model that has been modified to accept input as a sequence or a batch of 
sequences of 1D tensors (i.e. (batch_size, seq_len, state_dim) or (seq_len, state_dim)), and output 
the prediction of the next state only.
"""
class Transformer(torch.nn.Module):
    def __init__(self, feature_dim, N=3, d_model=128, d_ff=256, num_heads=8, dropout=0.1):
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
        
        self.predictor = ANN(d_model, [d_model * 2, d_model * 2], feature_dim, softmax=False)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0) # add batch dimension if there is none
        

        
"""The original transformer architecture that was not yet finished."""
# class Transformer(torch.nn.Module):
#     def __init__(self, feature_dim, N=3, d_model=128, d_ff=256, num_heads=8, dropout=0.1):
#         super(Transformer, self).__init__()
#         c = copy.deepcopy
#         attn = MultiHeadedAttention(num_heads, d_model)
#         ff = FeedForward(d_model, d_ff)
#         position = PositionalEncoding(d_model, dropout)
#         self.encoder_decoder = EncoderDecoder(
#             Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
#             torch.nn.Sequential(ANN(feature_dim, [d_model * 2, d_model * 2], d_model), c(position)),
#             torch.nn.Sequential(ANN(feature_dim, [d_model * 2, d_model * 2], d_model), c(position)),
#             Generator(d_model, feature_dim)
#         )
#         self.d_model = d_model
#         self.num_heads = num_heads

#         for p in self.encoder_decoder.parameters():
#             if p.dim() > 1:
#                 torch.nn.init.xavier_uniform_(p)

#     def get_src_mask(self, src):
#         batch_size = src.size(0)
#         seq_len = src.size(1)
#         src_mask = torch.ones(batch_size, self.num_heads, src.size(1), seq_len) # (batch_size, num_heads, seq_len, seq_len)
#         return src_mask
        
#     def get_memory(self, src, mask):
#         memory = self.encoder_decoder.encode(src, mask)
#         return memory
    
#     def forward(self, src):
#         if src.dim() == 2:
#             src = src.unsqueeze(0) # add batch dimension if there is none

#         src_mask = self.get_src_mask(src)
#         memory = self.get_memory(src, src_mask) # encode source


if __name__ == '__main__':
    def inference_test():
        test_model = Transformer(2)
        test_model.eval()
        src = torch.tensor([
            [[1,1], [2,1], [3,1], [4,1], [5,1], [6,1], [7,1], [8,1], [9,1], [10,1]],
            [[1,2], [2,2], [3,2], [4,2], [5,2], [6,2], [7,2], [8,2], [9,2], [10,2]],
            [[1,3], [2,3], [3,3], [4,3], [5,3], [6,3], [7,3], [8,3], [9,3], [10,3]]
        ], dtype=torch.float) # a batch of trajectories of 2 dimensional points

        print(test_model(src))

    for _ in range(5):
        inference_test()

