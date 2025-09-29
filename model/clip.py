import torch
from torch import nn 
from torch.nn import functional as F
import math 
from .attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self , n_vocab: int , n_embd: int , n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab , n_embd)
        self.position_embedding = nn.Parameter(torch.zeros((n_token , n_embd)))

    def forward(self, tokens):
        # (B , Seq_len) -> (B , Seq_len , Dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x 

class CLIPLayer(nn.Module):
    def __init__(self , n_embd: int , n_head : int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head , n_embd)
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd , 4 * n_embd) , 
            nn.GELU() , 
            nn.Linear(4 * n_embd , n_embd)
        )
    
    def forward(self , x):
        # x : (B , Seq_len , n_embd)
        residue = x 
        x = self.ln1(x)

        # x : (B , Seq_len , n_embd) reamains unchanged
        x = self.attention(x , causal_mask = True)

        x = x + residue

        residue = x 
        x = self.ln2(x)

        # x : (B , Seq_len , n_embd) reamains unchanged
        x = self.mlp(x)

        x += residue

        return x 


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding = CLIPEmbedding(
            n_vocab=49408,   # vocabulary size
            n_embd=768,      # embedding dimension
            n_token=77       # max sequence length
        )

        self.layers = nn.ModuleList([
            CLIPLayer(n_head=12, n_embd=768)  # 12 heads, 768-dim space
            for _ in range(12)                # stack 12 layers
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # Tokens -> embeddings
        # (B, Seq_len) -> (B, Seq_len, N_embd)
        state = self.embedding(tokens)

        # Pass through all transformer layers shapes unchanges
        for layer in self.layers:
            state = layer(state)

        # final output of shape (B , Seq_len , Dim = 768)
        output = self.layernorm(state)

        return output 
