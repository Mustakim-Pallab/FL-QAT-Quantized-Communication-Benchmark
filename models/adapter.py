import torch
import torch.nn as nn

from config import Config


class Adapter(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + residual


class AttentionAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, heads: int = 8, tokens: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"Attention adapter hidden dim ({hidden}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.memory_tokens = nn.Parameter(torch.empty(1, tokens, hidden))
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.up = nn.Linear(hidden, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.memory_tokens, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = x.shape
        x = x.view(batch, tokens, self.heads, self.head_dim)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, tokens, heads * head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.down(self.norm(x)).unsqueeze(1)
        memory = self.memory_tokens.expand(x.size(0), -1, -1)
        context = torch.cat([query, memory], dim=1)
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(context))
        v = self.split_heads(self.v_proj(context))
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        attended = torch.matmul(self.attn_dropout(attn), v)
        attended = self.out_proj(self.merge_heads(attended))
        adapted = attended + self.ffn(attended)
        residual = self.up(self.dropout(adapted.squeeze(1)))
        return x + residual


def build_adapter(dim: int, hidden: int, config: Config) -> nn.Module:
    if config.adapter_type == "attention":
        return AttentionAdapter(
            dim, hidden=hidden,
            heads=config.adapter_attention_heads,
            tokens=config.adapter_attention_tokens,
            dropout=config.adapter_dropout,
        )
    if config.adapter_type == "mlp":
        return Adapter(dim, hidden, dropout=config.adapter_dropout)
    raise ValueError(f"Unsupported adapter type: {config.adapter_type}")
