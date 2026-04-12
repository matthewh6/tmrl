import torch
import torch.nn as nn
from torch import Tensor

from tmrl.models.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from tmrl.models.common.utils import SinusoidalPosEmb, init_weights


class DualTimestepEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, mlp_ratio: float = 4.0):
        super().__init__()
        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t1, t2):
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        temb = torch.cat([temb1, temb2], dim=-1)
        return self.proj(temb)


class NoisePredNet(nn.Module):
    """Action noise prediction network."""

    def __init__(
        self,
        action_dim: int,
        action_len: int,
        global_cond_dim: int,
        context_dim: int = 0,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        # Action
        self.action_dim = action_dim
        self.action_len = action_len

        hidden_dim = embed_dim // 2
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Timestep
        if context_dim > 0:
            self.timestep_embedding = DualTimestepEncoder(embed_dim, mlp_ratio)
        else:
            self.timestep_embedding = SinusoidalPosEmb(embed_dim)

        # Global conditioning
        self.global_cond_embedding = nn.Sequential(
            nn.Linear(global_cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

        cond_dim = embed_dim * 2
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.pos_embed = nn.Parameter(torch.empty(1, action_len, embed_dim).normal_(std=0.02))

        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)
        self.action_inds = (0, action_len)

        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.initialize_weights()

    def forward(
        self, action: Tensor, t_action: Tensor, global_cond: Tensor, t_context: Tensor = None,
    ) -> Tensor:
        device = action.device

        # Action
        action_emb = self.action_embedding(action)
        x = action_emb + self.pos_embed

        # Timestep
        batch_size = action.shape[0]
        if len(t_action.shape) == 0:
            t_action = t_action.expand(batch_size).to(dtype=torch.long, device=device)
        t_action = t_action.view(-1).to(dtype=torch.long, device=device)
        if t_context is not None:
            if len(t_context.shape) == 0:
                t_context = t_context.expand(batch_size).to(dtype=torch.long, device=device)
            t_context = t_context.view(-1).to(dtype=torch.long, device=device)
            t_emb = self.timestep_embedding(t_context, t_action)
        else:
            t_emb = self.timestep_embedding(t_action)
            
        # Global conditioning
        global_cond_emb = self.global_cond_embedding(global_cond)
        cond = torch.cat((t_emb, global_cond_emb), dim=1)

        # Prediction
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)
        action_noise_pred = self.action_decoder(x[:, self.action_inds[0] : self.action_inds[1]])

        return action_noise_pred

    def initialize_weights(self):
        self.apply(init_weights)

        w = self.global_cond_embedding[0].weight.data
        nn.init.normal_(w.view([w.shape[0], -1]), mean=0.0, std=0.02)
        nn.init.constant_(self.global_cond_embedding[0].bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)
