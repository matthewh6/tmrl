import torch
from torch import Tensor
from typing import Dict
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from tmrl.utils.logging import cprint
from tmrl.models.common.utils import SinusoidalPosEmb
import itertools
import time

log = lambda msg, color='bright_green': cprint(msg, color)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


def create_optimizers(model: object, cfg: object) -> dict[str, torch.optim.Optimizer]:
    optimizers = {
        'qf': torch.optim.Adam(
            itertools.chain(*[q.parameters() for q in model.critic_ensemble]),
            lr=cfg.optimizer.lr,
        ),
        'actor': torch.optim.Adam(model.actor.parameters(), lr=cfg.optimizer.lr),
    }

    if cfg.use_autotune:
        optimizers['alpha_a'] = torch.optim.Adam([model.log_alpha_a], lr=cfg.optimizer.lr)
        if model.method == 'tmrl':
            optimizers['alpha_t'] = torch.optim.Adam([model.log_alpha_t], lr=cfg.optimizer.lr)
            optimizers['alpha_c'] = torch.optim.Adam([model.log_alpha_c], lr=cfg.optimizer.lr)

    return optimizers


class SACAgent(nn.Module):
    def __init__(
        self,
        method: str,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        action_len: int,
        context_dim: int,
        noise_bound: float,
        context_noise_bound: float,
        timestep_bound: float = 1.0,
        hidden_dim: int = 256,
        n_critics: int = 5,
        n_layers: int = 3,
        use_autotune: bool = True,
        discount: float = 0.99,
        tau: float = 0.005,
        critic_updates_per_actor_update: int = 1,
    ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.method = method

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.action_len = action_len
        self.action_len = 1  # TODO: action noise
        self.context_dim = context_dim

        self.noise_bound = noise_bound
        self.context_noise_bound = context_noise_bound
        self.timestep_bound = timestep_bound

        self.hidden_dim = hidden_dim
        self.n_critics = n_critics
        self.n_layers = n_layers

        self.use_autotune = use_autotune
        self.discount = discount
        self.tau = tau

        self.critic_updates_per_actor_update = critic_updates_per_actor_update

        log(f'discount: {self.discount}')
        log(f'action_len: {self.action_len}')

    def initialize(self) -> None:
        """Initialize entropy, actor, and critic networks"""

        # Entropy
        if self.use_autotune:
            self.target_entropy_a = -float(self.action_dim)
            self.target_entropy_t = -float(1)
            self.target_entropy_c = -float(self.context_dim)

            log(f'target_entropy_a: {self.target_entropy_a}')
            log(f'target_entropy_t: {self.target_entropy_t}')
            log(f'target_entropy_c: {self.target_entropy_c}')

            self.log_alpha_a = nn.Parameter(torch.zeros(1, device=self.device))
            self.log_alpha_t = nn.Parameter(torch.zeros(1, device=self.device))
            self.log_alpha_c = nn.Parameter(torch.zeros(1, device=self.device))

        else:
            self.alpha_a = 0.2
            self.alpha_t = 0.2
            self.alpha_c = 0.2

        # Actor network
        self.actor = Actor(
            method=self.method,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            noise_bound=self.noise_bound,
            context_noise_bound=self.context_noise_bound,
            timestep_bound=self.timestep_bound,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
        )

        # Noise critic ensemble Q(s,z,g)
        self.critic_ensemble = nn.ModuleList(
            [
                SoftQNetwork(
                    method=self.method,
                    obs_dim=self.obs_dim,
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    action_len=self.action_len,
                    hidden_dim=self.hidden_dim,
                    n_layers=self.n_layers,
                    context_dim=self.context_dim,
                )
                for _ in range(self.n_critics)
            ]
        )

        self.target_critic_ensemble = nn.ModuleList([deepcopy(critic) for critic in self.critic_ensemble])

    def compute_critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        timesteps: Tensor | None,
        context_noises: Tensor | None,
        next_obs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        log_dict: dict[str, object] | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        if log_dict is None:
            log_dict = {}
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.get_action(next_obs)
            next_actions, next_timesteps, next_context_noises = next_actions
            next_log_prob_a, next_log_prob_t, next_log_prob_p = next_log_prob
            next_actions = next_actions.unsqueeze(1).repeat(1, self.action_len, 1)

            next_q_targets = self.get_target_q(
                obs=next_obs,
                actions=next_actions,
                timesteps=next_timesteps,
                context_noises=next_context_noises,
            )

            if self.use_autotune:
                alpha_a = self.log_alpha_a.detach().exp()
                alpha_t = self.log_alpha_t.detach().exp()
                alpha_c = self.log_alpha_c.detach().exp()
            else:
                alpha_a, alpha_t, alpha_c = self.alpha_a, self.alpha_t, self.alpha_c

            entropy_bonus = alpha_a * next_log_prob_a
            if self.method == 'tmrl':
                entropy_bonus = entropy_bonus + alpha_t * next_log_prob_t + alpha_c * next_log_prob_p

            next_q_target = next_q_targets.min(dim=0).values
            target_q = rewards + (1 - terminals) * (self.discount**self.action_len) * next_q_target

        # Only use the first action from the chunk for Q(s_t, a_t)
        actions_q = actions[:, : self.action_len, :]
        qs = self.get_q(obs=obs, actions=actions_q, timesteps=timesteps, context_noises=context_noises)
        target_q_expanded = target_q.unsqueeze(0).expand_as(qs)  # [N, B, 1]
        critic_loss = F.mse_loss(qs, target_q_expanded)
        log_dict.update(
            {
                'critic/loss': critic_loss,
                'critic/target_q': target_q.mean(),
                'critic/target_q_min': target_q.min(),
                'critic/target_q_max': target_q.max(),
                'critic/q': qs.mean(),
                'critic/q_min': qs.min(),
                'critic/q_max': qs.max(),
                'critic/rewards': rewards.mean(),
            }
        )

        return critic_loss, log_dict

    def compute_actor_and_alpha_loss(
        self,
        obs: Tensor,
        log_dict: dict[str, object] | None = None,
    ) -> tuple[Tensor, Tensor | float | None, Tensor | float | None, Tensor | float | None, dict[str, object]]:
        if log_dict is None:
            log_dict = {}
        actions, log_prob, _ = self.actor.get_action(obs)
        actions, timesteps, context_noises = actions
        actions = actions.unsqueeze(1).repeat(1, self.action_len, 1)
        log_prob_a, log_prob_t, log_prob_p = log_prob

        qs = self.get_q(
            obs=obs,
            actions=actions,
            timesteps=timesteps,
            context_noises=context_noises,
        )

        q = qs.min(dim=0)[0]

        # SAC entropy bonus
        if self.use_autotune:
            alpha_a = self.log_alpha_a.detach().exp()
            alpha_t = self.log_alpha_t.detach().exp()
            alpha_c = self.log_alpha_c.detach().exp()

            if self.method == 'tmrl':
                actor_loss = ((alpha_a * log_prob_a) + (alpha_t * log_prob_t) + (alpha_c * log_prob_p) - q).mean()
            else:
                actor_loss = (alpha_a * log_prob_a - q).mean()

            alpha_a_loss = -(self.log_alpha_a * (log_prob_a + self.target_entropy_a).detach()).mean()
            if self.method == 'tmrl':
                alpha_t_loss = -(self.log_alpha_t * (log_prob_t + self.target_entropy_t).detach()).mean()
                alpha_c_loss = -(self.log_alpha_c * (log_prob_p + self.target_entropy_c).detach()).mean()

                log_dict.update(
                    {
                        'actor/alpha_t': alpha_t,
                        'actor/alpha_c': alpha_c,
                        'actor/alpha_t_loss': alpha_t_loss,
                        'actor/alpha_c_loss': alpha_c_loss,
                    }
                )
            else:
                alpha_t_loss = 0.0
                alpha_c_loss = 0.0

            log_dict.update(
                {
                    'actor/alpha_a': alpha_a,
                    'actor/alpha_a_loss': alpha_a_loss,
                }
            )

        else:
            alpha_a = self.alpha_a
            alpha_t = self.alpha_t
            alpha_c = self.alpha_c

            if self.method == 'tmrl':
                actor_loss = ((alpha_a * log_prob_a) + (alpha_t * log_prob_t) + (alpha_c * log_prob_p) - q).mean()
            else:
                actor_loss = (alpha_a * log_prob_a - q).mean()

            alpha_a_loss = None
            alpha_t_loss = None
            alpha_c_loss = None

        log_dict.update(
            {
                'actor/actor_loss': actor_loss,
                'actor/q': qs.mean(),
                'actor/q_min': qs.min(),
                'actor/q_max': qs.max(),
                'actor/action_min': actions.min(),
                'actor/action_mean': actions.mean(),
                'actor/action_max': actions.max(),
                'actor/log_prob_a': log_prob_a.mean(),
            }
        )

        if self.method == 'tmrl':
            log_dict.update(
                {
                    'actor/timestep_min': timesteps.min(),
                    'actor/timestep_mean': timesteps.mean(),
                    'actor/timestep_max': timesteps.max(),
                    'actor/context_noise_min': context_noises.min(),
                    'actor/context_noise_mean': context_noises.mean(),
                    'actor/context_noise_max': context_noises.max(),
                    'actor/log_prob_t': log_prob_t.mean(),
                    'actor/log_prob_p': log_prob_p.mean(),
                }
            )

        return actor_loss, alpha_a_loss, alpha_t_loss, alpha_c_loss, log_dict

    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        timesteps: Tensor | None,
        context_noises: Tensor | None,
        next_obs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        optimizers: dict[str, torch.optim.Optimizer] | None = None,
    ) -> Dict[str, object]:
        log_dict = {}

        def _slice_batch(x: object, idx: Tensor) -> object:
            if x is None:
                return None
            if isinstance(x, dict):
                return {k: _slice_batch(v, idx) for k, v in x.items()}
            return x[idx]

        B = actions.shape[0]

        # Each update gets its own random subset of the batch
        all_indices = [
            torch.randperm(B, device=actions.device)[: B // self.critic_updates_per_actor_update]
            for _ in range(self.critic_updates_per_actor_update)
        ]

        # --- Critic updates (N per actor update) ---
        for critic_update in range(self.critic_updates_per_actor_update):
            start_time = time.time()
            idx = all_indices[critic_update]

            critic_loss, log_dict = self.compute_critic_loss(
                obs=_slice_batch(obs, idx),
                actions=_slice_batch(actions, idx),
                timesteps=_slice_batch(timesteps, idx),
                context_noises=_slice_batch(context_noises, idx),
                next_obs=_slice_batch(next_obs, idx),
                rewards=_slice_batch(rewards, idx),
                terminals=_slice_batch(terminals, idx),
                log_dict=log_dict,
            )
            optimizers['qf'].zero_grad()
            critic_loss.backward()
            optimizers['qf'].step()

            log_dict['critic/time'] = time.time() - start_time

        self.update_target_networks()

        # --- Actor update ---
        actor_idx = torch.randperm(B, device=actions.device)[
            : B // self.critic_updates_per_actor_update
        ]  # use a fresh randperm

        actor_loss, alpha_a_loss, alpha_t_loss, alpha_c_loss, log_dict = self.compute_actor_and_alpha_loss(
            obs=_slice_batch(obs, actor_idx),
            log_dict=log_dict,
        )
        optimizers['actor'].zero_grad()
        actor_loss.backward()
        optimizers['actor'].step()

        if self.use_autotune:
            optimizers['alpha_a'].zero_grad()
            alpha_a_loss.backward()
            optimizers['alpha_a'].step()

            if self.method == 'tmrl':
                optimizers['alpha_t'].zero_grad()
                alpha_t_loss.backward()
                optimizers['alpha_t'].step()
                optimizers['alpha_c'].zero_grad()
                alpha_c_loss.backward()
                optimizers['alpha_c'].step()

        return log_dict

    def get_q(
        self,
        obs: Tensor,
        actions: Tensor,
        timesteps: Tensor | None = None,
        context_noises: Tensor | None = None,
    ) -> Tensor:
        """Get Q-values from the noise Q-function networks"""

        qs = torch.stack(
            [q(obs, actions=actions, timesteps=timesteps, context_noises=context_noises) for q in self.critic_ensemble]
        )  # [N, B, 1]

        return qs

    def get_target_q(
        self,
        obs: Tensor,
        actions: Tensor,
        timesteps: Tensor | None = None,
        context_noises: Tensor | None = None,
    ) -> Tensor:
        """Get target Q-values from the target Q-function networks"""

        qs = torch.stack(
            [
                q(obs, actions=actions, timesteps=timesteps, context_noises=context_noises)
                for q in self.target_critic_ensemble
            ]
        )  # [N, B, 1]

        return qs

    @torch.no_grad()
    def update_target_networks(self) -> None:
        tau = self.tau
        one_minus_tau = 1 - tau

        for critic, target_critic in zip(self.critic_ensemble, self.target_critic_ensemble):
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.copy_(one_minus_tau * target_param + tau * param)


class Actor(nn.Module):
    def __init__(
        self,
        method: str,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        noise_bound: float = 1.0,
        context_noise_bound: float = 1.0,
        timestep_bound: float = 1.0,
        context_dim: int = 2048,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.method = method

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        input_dim = self.obs_dim + self.state_dim

        # Shared trunk
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Noise action head
        self.mu_action = nn.Linear(hidden_dim, action_dim)
        self.log_std_action = nn.Linear(hidden_dim, action_dim)

        # Timestep action head
        if self.method == 'tmrl':
            self.mu_timestep = nn.Linear(hidden_dim, 1)
            self.log_std_timestep = nn.Linear(hidden_dim, 1)

            self.mu_context_noise = nn.Linear(hidden_dim, context_dim)
            self.log_std_context_noise = nn.Linear(hidden_dim, context_dim)

        # Scaling factors
        self.register_buffer('action_scale', torch.tensor(noise_bound, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor(0.0, dtype=torch.float32))

        self.register_buffer('timestep_scale', torch.tensor(timestep_bound, dtype=torch.float32))
        self.register_buffer('timestep_bias', torch.tensor(0.0, dtype=torch.float32))

        self.register_buffer('context_scale', torch.tensor(context_noise_bound, dtype=torch.float32))
        self.register_buffer('context_bias', torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self, x: Tensor
    ) -> tuple[
        tuple[Tensor, Tensor | None, Tensor | None],
        tuple[Tensor, Tensor | None, Tensor | None],
    ]:
        h = self.net(x)

        # Noise action distribution
        mu_a = self.mu_action(h)
        log_std_a = torch.tanh(self.log_std_action(h))
        log_std_a = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_a + 1)

        # Timestep distribution
        if self.method == 'tmrl':
            mu_t = self.mu_timestep(h)
            log_std_t = torch.tanh(self.log_std_timestep(h))
            log_std_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_t + 1)

            mu_p = self.mu_context_noise(h)
            log_std_p = torch.tanh(self.log_std_context_noise(h))
            log_std_p = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_p + 1)
        else:
            mu_t = None
            log_std_t = None
            mu_p = None
            log_std_p = None

        return (mu_a, mu_t, mu_p), (log_std_a, log_std_t, log_std_p)

    def get_action(
        self,
        obs: Tensor,
    ) -> tuple[
        tuple[Tensor, Tensor | None, Tensor | None],
        tuple[Tensor, Tensor | None, Tensor | None],
        tuple[Tensor, Tensor | None, Tensor | None],
    ]:
        """Sample actions + compute log-probs."""
        (mu_a, mu_t, mu_p), (log_std_a, log_std_t, log_std_p) = self(obs)

        # --- Action noise ---
        std_a = log_std_a.exp()
        normal_a = torch.distributions.Normal(mu_a, std_a)
        x_a = normal_a.rsample()  # reparameterization trick
        y_a = torch.tanh(x_a)
        actions_a = y_a * self.action_scale + self.action_bias

        # --- Timestep ---
        if self.method == 'tmrl':
            std_t = log_std_t.exp()
            normal_t = torch.distributions.Normal(mu_t, std_t)
            x_t = normal_t.rsample()
            y_t = torch.tanh(x_t)
            actions_t = y_t * self.timestep_scale + self.timestep_bias

            std_p = log_std_p.exp()
            normal_p = torch.distributions.Normal(mu_p, std_p)
            x_p = normal_p.rsample()
            y_p = torch.tanh(x_p)
            actions_p = y_p * self.context_scale + self.context_bias
        else:
            actions_t = None
            actions_p = None

        # Log-prob (mean over dims so entropy scale is dimension-independent)
        log_prob_a = normal_a.log_prob(x_a)
        log_prob_a -= torch.log(self.action_scale * (1 - y_a.pow(2)) + 1e-6)
        # log_prob_a = log_prob_a.mean(dim=1, keepdim=True)
        log_prob_a = log_prob_a.sum(dim=1, keepdim=True)

        if self.method == 'tmrl':
            log_prob_t = normal_t.log_prob(x_t)
            log_prob_t -= torch.log(self.timestep_scale * (1 - y_t.pow(2)) + 1e-6)
            # log_prob_t = log_prob_t.mean(dim=1, keepdim=True)
            log_prob_t = log_prob_t.sum(dim=1, keepdim=True)

            log_prob_p = normal_p.log_prob(x_p)
            log_prob_p -= torch.log(self.context_scale * (1 - y_p.pow(2)) + 1e-6)
            # log_prob_p = log_prob_p.mean(dim=1, keepdim=True)
            log_prob_p = log_prob_p.sum(dim=1, keepdim=True)
        else:
            log_prob_t = None
            log_prob_p = None

        mu_a = torch.tanh(mu_a) * self.action_scale + self.action_bias
        if self.method == 'tmrl':
            mu_t = torch.tanh(mu_t) * self.timestep_scale + self.timestep_bias
            mu_p = torch.tanh(mu_p) * self.context_scale + self.context_bias

        return (actions_a, actions_t, actions_p), (log_prob_a, log_prob_t, log_prob_p), (mu_a, mu_t, mu_p)


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        method: str,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        action_len: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        context_dim: int = 2048,
        num_train_timesteps: int = 100,  # TODO: set this based on the checkpoint
    ) -> None:
        super().__init__()
        self.method = method

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_len = action_len

        input_dim = self.obs_dim + self.state_dim + action_dim * action_len

        self.num_train_timesteps = num_train_timesteps
        if self.method == 'tmrl':
            self.timestep_embedding = SinusoidalPosEmb(64)
            input_dim += 64 + context_dim
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()])
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs: Tensor,
        actions: Tensor,
        timesteps: Tensor | None = None,
        context_noises: Tensor | None = None,
    ) -> Tensor:

        # Flatten inputs to a vector
        actions = actions.view(actions.shape[0], -1)
        inputs = torch.cat([obs, actions], dim=-1)

        if self.method == 'tmrl':
            assert timesteps.min() >= -1 and timesteps.max() <= 1, (
                f'Timesteps should be in the range of (-1, 1), but got {timesteps.min()} and {timesteps.max()}'
            )
            timesteps = (timesteps + 1) / 2
            timesteps = timesteps * self.num_train_timesteps

            t_emb = self.timestep_embedding(timesteps.squeeze(-1))

            inputs = torch.cat([inputs, t_emb, context_noises], dim=-1)

        q = self.net(inputs)

        return q
