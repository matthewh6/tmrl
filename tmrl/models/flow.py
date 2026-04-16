import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tmrl.models.common.noise import NoisePredNet


class FlowPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_len: int,
        action_dim: int,
        goal_dim: int = 0,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        timeshift: float = 1.0,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_len = action_len
        self.action_dim = action_dim
        self.action_shape = (action_len, action_dim)
        self.goal_dim = goal_dim

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps

        timesteps = torch.linspace(1, 0, self.num_inference_steps + 1)
        self.register_buffer('timesteps', (timeshift * timesteps) / (1 + (timeshift - 1) * timesteps))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.noise_pred_net = NoisePredNet(
            action_dim=action_dim,
            action_len=action_len,
            global_cond_dim=obs_dim + goal_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.rand((batch_size,), device=device)

    @torch.no_grad()
    def sample(
        self, obs: Tensor, goals: Tensor | None = None, action_noise: Tensor | None = None, **kwargs: object
    ) -> Tensor:
        batch_size, device = obs.shape[0], obs.device

        if action_noise is None:
            action_sample = torch.randn((batch_size, *self.action_shape), device=device)
        else:
            if action_noise.ndim == 2:
                action_noise = action_noise.unsqueeze(1).expand(-1, self.action_len, -1)
            action_sample = action_noise

        global_cond = torch.cat([obs, goals], dim=-1) if self.goal_dim > 0 else obs

        for tcont_action, tcont_next_action in zip(self.timesteps[:-1], self.timesteps[1:]):
            t_action = (tcont_action * self.num_train_steps).long()
            flow_pred = self.noise_pred_net(
                action=action_sample, t_action=t_action, global_cond=global_cond,
            )
            action_sample = action_sample + (tcont_next_action - tcont_action) * flow_pred

        return action_sample

    def forward(
        self, obs: Tensor, actions: Tensor, goals: Tensor | None = None, **kwargs: object
    ) -> tuple[Tensor, dict[str, float]]:
        batch_size, device = obs.shape[0], obs.device

        action_noise = torch.randn_like(actions)
        tcont_action = self.sample_timesteps(batch_size, device)
        t_action = (tcont_action * self.num_train_steps).long()

        direction_action = action_noise - actions
        noisy_action = actions + tcont_action.view(-1, 1, 1) * direction_action
        global_cond = torch.cat([obs, goals], dim=-1) if self.goal_dim > 0 else obs

        noise_pred_action = self.noise_pred_net(
            action=noisy_action, t_action=t_action, global_cond=global_cond,
        )

        loss = F.mse_loss(noise_pred_action, direction_action)
        return loss, {'loss': loss.item()}


class ContextSmoothedFlowPolicy(FlowPolicy):
    """Unified context smoothed flow policy.

    Determines what to noise based on context_dim:
      - context_dim == goal_dim noises the goal
      - context_dim == state_dim: noises the observation/state
    """

    def __init__(
        self,
        obs_dim: int,
        action_len: int,
        action_dim: int,
        goal_dim: int,
        context_dim: int,
        num_train_steps: int = 100,
        num_inference_steps: int = 5,
        timeshift: float = 1.0,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            action_len=action_len,
            action_dim=action_dim,
            goal_dim=goal_dim,
            num_train_steps=num_train_steps,
            num_inference_steps=num_inference_steps,
            timeshift=timeshift,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.context_dim = context_dim
        # self.num_context_steps = num_context_steps # TODO: add

        self.noise_goal = context_dim == goal_dim
        self.noise_state = context_dim == obs_dim

        assert self.noise_goal or self.noise_state, "ContextSmoothedFlowPolicy must noise either the goal or the state"

        # Override with dual-timestep net for context noising
        self.noise_pred_net = NoisePredNet(
            action_dim=action_dim,
            action_len=action_len,
            global_cond_dim=obs_dim + goal_dim if self.noise_goal else obs_dim,
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
        )

        betas = torch.linspace(1e-4, 0.02, num_train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer('alpha_bars', torch.cumprod(alphas, dim=0))

    def _apply_diffusion_noise(self, x: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """x_t = sqrt(alpha_bar_t) * x + sqrt(1 - alpha_bar_t) * noise"""
        ab_t = self.alpha_bars[t]
        view = (-1,) + (1,) * (x.dim() - 1)
        return torch.sqrt(ab_t).view(view) * x + torch.sqrt(1.0 - ab_t).view(view) * noise

    @torch.no_grad()
    def sample(
        self,
        obs: Tensor,
        goals: Tensor | None = None,
        tcont_context: Tensor | None = None,
        action_noise: Tensor | None = None,
        context_noise: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        batch_size, device = obs.shape[0], obs.device

        if action_noise is None:
            action_sample = torch.randn((batch_size, *self.action_shape), device=device)
        else:
            if action_noise.ndim == 2:
                action_noise = action_noise.unsqueeze(1).expand(-1, self.action_len, -1)
            action_sample = action_noise

        # Continuous -> Discrete timestep
        if tcont_context is None:
            tcont_context = torch.ones((batch_size,), device=device) * 0.999
        else:
            if tcont_context.ndim == 1:
                tcont_context = tcont_context.expand(batch_size,)
            tcont_context = tcont_context.to(dtype=torch.float, device=device)
        t_context = (tcont_context * self.num_train_steps).long().clamp(max=self.num_train_steps - 1)

        if context_noise is None:
            context_noise = torch.randn((batch_size, self.context_dim), device=device)

        if self.noise_goal:
            noisy_context = self._apply_diffusion_noise(goals, t_context, context_noise)
            global_cond = torch.cat([obs, noisy_context], dim=-1)
        else:
            noisy_context = self._apply_diffusion_noise(obs, t_context, context_noise)
            global_cond = torch.cat([noisy_context], dim=-1)

        for tcont_action, tcont_next_action in zip(self.timesteps[:-1], self.timesteps[1:]):
            t_action = (tcont_action * self.num_train_steps).long()
            noise_pred_action = self.noise_pred_net(
                action=action_sample,
                t_action=t_action,
                global_cond=global_cond,
                t_context=t_context,
            )
            action_sample = action_sample + (tcont_next_action - tcont_action) * noise_pred_action

        return action_sample

    def forward(
        self,
        obs: Tensor,
        actions: Tensor,
        goals: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, dict[str, float]]:
        batch_size, device = obs.shape[0], obs.device

        context_noise = torch.randn((batch_size, self.context_dim), device=device)
        action_noise = torch.randn_like(actions)

        tcont_context = self.sample_timesteps(batch_size, device)
        tcont_action = self.sample_timesteps(batch_size, device)

        t_context = (tcont_context * self.num_train_steps).long().clamp(max=self.num_train_steps - 1)
        t_action = (tcont_action * self.num_train_steps).long()

        if self.noise_goal:
            noisy_context = self._apply_diffusion_noise(goals, t_context, context_noise)
            global_cond = torch.cat([obs, noisy_context], dim=-1)
        else:
            noisy_context = self._apply_diffusion_noise(obs, t_context, context_noise)
            global_cond = torch.cat([noisy_context], dim=-1)

        direction_action = action_noise - actions
        noisy_action = actions + tcont_action.view(-1, 1, 1) * direction_action

        noise_pred_action = self.noise_pred_net(
            action=noisy_action,
            t_action=t_action,
            global_cond=global_cond,
            t_context=t_context,
        )

        loss = F.mse_loss(noise_pred_action, direction_action)
        info = {'loss': loss.item()}
        return loss, info
