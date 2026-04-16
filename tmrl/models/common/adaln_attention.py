import torch.nn as nn
from torch import Tensor

from .attention import MLP, Attention, CrossAttention


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNAttentionBlock(nn.Module):
    """Multiheaded self-attention block with adaptive layer normalization modulation."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act: type[nn.Module] = nn.GELU,
        norm: type[nn.Module] = nn.LayerNorm,
        is_causal: bool = False,
        causal_block: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            causal_block=causal_block,
        )
        self.norm2 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=act,
            drop=drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )

    def forward(
        self, x: Tensor, cond: Tensor, pos_embed: Tensor | None = None, attn_mask: Tensor | None = None
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos_embed, attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class AdaLNCrossAttentionBlock(nn.Module):
    """Multiheaded cross-attention block with adaptive layer normalization modulation."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act: type[nn.Module] = nn.GELU,
        norm: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.xattn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=act,
            drop=drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )

    def forward(
        self, x: Tensor, c: Tensor, cond: Tensor, x_pos_embed: Tensor | None = None, c_pos_embed: Tensor | None = None
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.xattn(
            modulate(x, shift_msa, scale_msa), self.norm1(c), x_pos_embed, c_pos_embed
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class AdaLNHybridAttentionBlock(nn.Module):
    """Multiheaded hybrid attention block with adaptive layer normalization modulation."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act: type[nn.Module] = nn.GELU,
        norm: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.xattn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm3 = norm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=act,
            drop=drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )

    def forward(
        self, x: Tensor, c: Tensor, cond: Tensor, x_pos_embed: Tensor | None = None, c_pos_embed: Tensor | None = None
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), x_pos_embed)
        x = x + self.xattn(self.norm2(x), c, x_pos_embed, c_pos_embed)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class AdaLNFinalLayer(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = self.linear(modulate(self.norm(x), shift, scale))
        return x
