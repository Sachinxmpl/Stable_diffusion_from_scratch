import torch
from torch import nn 
from torch.nn import functional as F
import math 

class SelfAttention(nn.Module):
    def __init__(self , num_heads , embed_dim):
        """
        Transformer multi-head attention 
        """
        super().__init__()

        assert embed_dim % num_heads == 0 , "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim , embed_dim)
        self.k_linear = nn.Linear(embed_dim , embed_dim)
        self.v_linear = nn.Linear(embed_dim , embed_dim)

        self.out_proj = nn.Linear(embed_dim , embed_dim)

        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

    def forward(self , x, causal_mask = False):
        # x: (batch , seq_len , embed_dim)
        batch , seq_len , embed_dim = x.shape

        # computer Q , K , V 
        Q = self.q_linear(x)  # (batch , seq_len , embed_dim)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # split into head 
        # (batch , seq_len , embed_dim) --> (batch , seq_len , num_head , head_dim).T --> (batch , num_heads , seq_len , head_dim)
        Q = Q.view(batch , seq_len , self.num_heads , self.head_dim).transpose(1,2)
        K = K.view(batch , seq_len , self.num_heads , self.head_dim).transpose(1,2)
        V = V.view(batch , seq_len , self.num_heads , self.head_dim).transpose(1,2)

        # scores of dim (batch ,num_heads , seq_len , seq_len)
        scores = torch.matmul(Q , K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # add masking 
        if causal_mask:
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)   # upper triangle True
            scores = scores.masked_fill(mask, float("-inf"))


        attn = F.softmax(scores , dim = -1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn , V)

        # merge heads -> (batch , seq_len , embed_dim)
        context = context.transpose(1,2).contiguous().view(batch , seq_len ,  self.embed_dim)
        out = self.out_proj(context)
        out = self.proj_dropout(out)
        
        return out
    

class CrossAttention(nn.Module):
    def __init__(self , embed_dim , cross_dim , num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # query comes from x , key and value comes from y
        self.q_linear = nn.Linear(embed_dim , embed_dim)
        self.k_linear = nn.Linear(cross_dim , embed_dim)
        self.v_linear = nn.Linear(cross_dim , embed_dim)

        self.out_proj = nn.Linear(embed_dim , embed_dim)

    def forward(self , x, y):
        # x : (Batch_size , seq_len_x , embed_dim)  latent sequences
        # y : (Batch_size , seq_len_y , cross_dim)  text tokens
        batch_size , seq_len_x , _ = x.shape
        _ , seq_len_y , _ = y.shape

        Q = self.q_linear(x)
        K = self.k_linear(y)
        V = self.v_linear(y)

        # split to multiples heads
        Q = Q.view(batch_size , seq_len_x , self.num_heads , self.head_dim).transpose(1,2)
        K = K.view(batch_size , seq_len_y , self.num_heads , self.head_dim).transpose(1,2)
        V = V.view(batch_size , seq_len_y , self.num_heads , self.head_dim).transpose(1 ,2)

        # Attention 
        scores = torch.matmul(Q , K.transpose(-2 , -1)) /  math.sqrt(self.head_dim)
        attn = F.softmax(scores , dim = -1)
        context  = torch.matmul(attn , V)

        # merge heads 
        context = context.transpose(1 ,2).contiguous().view(batch_size , seq_len_x , self.embed_dim 
                                                            )
        out = self.out_proj(context)
        return out 
