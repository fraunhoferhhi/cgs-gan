import torch
import torch.nn as nn
from torch_utils import persistence
import math
import numpy as np
from training.networks_stylegan2 import FullyConnectedLayer


@persistence.persistent_class
class InstanceNorm1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True, unbiased=False) + 1e-8)
        return x.to(dtype)
        
    
@persistence.persistent_class
class AdaptiveNorm(nn.Module):
    def __init__(
        self,
        dim,
        w_dim,
        weight_init=1.0,
    ):
        super().__init__()

        self.gamma = FullyConnectedLayer(w_dim, dim, activation='linear', weight_init=weight_init, bias_init=1.)
        self.beta = FullyConnectedLayer(w_dim, dim, activation='linear', weight_init=weight_init, bias_init=0.)
        self.norm = InstanceNorm1d()

    def forward(self, x, w):
        if len(w.shape) == 2:
            return self.norm(x) * self.gamma(w).unsqueeze(1) + self.beta(w).unsqueeze(1)
        else:
            return self.norm(x) * self.gamma(w) + self.beta(w)

@persistence.persistent_class
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        
        self.attention = QKVMultiheadAttention(heads=heads)
        self.c_qkv = FullyConnectedLayer(width, width * 3)
        self.c_proj = FullyConnectedLayer(width, width)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


@persistence.persistent_class
class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.out_channels = width
        self.gelu = nn.GELU()
        self.c_fc = FullyConnectedLayer(width, width * 4)
        self.c_proj = FullyConnectedLayer(width * 4, width)

    def forward(self, x, w=None):
        return self.c_proj(self.gelu(self.c_fc(x)) * np.sqrt(2))


@persistence.persistent_class
class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int):
        super().__init__()
        self.heads = heads

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        
        # Dot product attention
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


@persistence.persistent_class
class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        w_dim,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            width=width,
            heads=heads,
        )
        self.mlp = MLP(width=width)
        self.ln_1 = AdaptiveNorm(width, w_dim=w_dim)
        self.ln_2 = AdaptiveNorm(width, w_dim=w_dim)
        self.ls_1 = FullyConnectedLayer(w_dim, width, activation='linear', weight_init=0.)
        self.ls_2 = FullyConnectedLayer(w_dim, width, activation='linear', weight_init=0.)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = x + self.attn(self.ln_1(x, w)) * self.ls_1(w)
        x = x + self.mlp(self.ln_2(x, w)) * self.ls_2(w)
        return x


@persistence.persistent_class
class Transformer(nn.Module):
    def __init__(
            self,
            *,
            w_dim,
            width: int,
            layers: int,
            num_first_layers=6,
            heads: int = 8,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width=width,
                heads=heads,
                w_dim=w_dim
            )
            for i in range(layers - 1)
        ])

        self.first_layers = nn.ModuleList([
            ResidualAttentionBlock(
                width=width,
                heads=heads,
                w_dim=w_dim
            )
            for i in range(num_first_layers)
        ])
        self.num_first_layers = num_first_layers

    def forward(self, x: torch.Tensor, ws: torch.Tensor):
        results = []

        for i in range(self.num_first_layers):
            x = self.first_layers[i](x, ws)
        results.append(x)

        for i in range(len(self.resblocks)):
            x = self.resblocks[i](x, ws)
            results.append(x)
        return results
