import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)


@dataclass
class Config:
    hidden_size: int = 4096
    num_heads: int = 16 
    head_dim: int = 256

    kv_lora_rank: int = 512
    qk_rope_dim: int = 512 # Decoupled rope


class MultiLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads 
        self.head_dim = config.head_dim
        
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_dim = config.qk_rope_dim 

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * (self.head_dim + self.qk_rope_dim))

        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)

        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.head_dim + self.qk_rope_dim))

    def forward(self, x, latent_cache: list = None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, -1)
        
        q_content, q_rope = torch.split(q, [self.head_dim, self.qk_rope_dim], dim=-1)
        
        kv_latent = self.kv_down_proj(x)
        kv_latent = self.kv_norm(kv_latent)

        kv = self.kv_up_proj(kv_latent)
        kv = kv.view(batch_size, seq_len, self.num_heads, -1)
        k_content, k_rope, v = torch.split(kv, [self.head_dim, self.qk_rope_dim, self.head_dim], dim=-1)


def weight_absorbtion():
    batch_size = 32
    seq_len = 1024
    hidden_dim = 4096
    latent_dim = 64
    head_dim = 256

    x = torch.randn(batch_size, seq_len, hidden_dim)
    c_j = torch.randn(batch_size, seq_len, latent_dim)

    W_Q = torch.randn(hidden_dim, head_dim)
    W_UK = torch.randn(latent_dim, head_dim)

    q = torch.matmul(x, W_Q)
    k = torch.matmul(c_j, W_UK)

    score_standard = torch.matmul(q, k.transpose(-1, -2))

    W_absorbed = torch.matmul(W_Q, W_UK.t())

    q_latent = torch.matmul(x, W_absorbed)

    score_absorbed = torch.matmul(q_latent, c_j.transpose(-1, -2))

    diff = torch.abs(score_absorbed - score_standard).max().item()

    print(f"{diff = }")

if __name__ == "__main__":
    weight_absorbtion()
