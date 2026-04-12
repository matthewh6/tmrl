import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tmrl.utils.logging import cprint
import tqdm
import wandb


log = lambda msg, color='bright_green': cprint(msg, color)


class EnsembleMember(nn.Module):
    """Simple MLP that predicts actions from observations."""

    def __init__(self, obs_dim, action_dim, hidden_size=512, num_layers=3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class PosteriorVarianceEstimator:
    """Fits an ensemble on bootstrapped data to estimate per-state posterior
    covariance of the demonstrator's actions.

    Usage:
        estimator = PosteriorVarianceEstimator(obs_dim=4, action_dim=2)
        estimator.fit(obs_np, actions_np)
        var = estimator.compute_covariance(obs_tensor)  # [N, action_dim]
    """

    def __init__(self, obs_dim, action_dim, ensemble_size=10,
                 hidden_size=512, num_layers=3, device='cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.device = device
        self.members = [
            EnsembleMember(obs_dim, action_dim, hidden_size, num_layers).to(device)
            for _ in range(ensemble_size)
        ]

    def fit(self, obs, actions, num_epochs=3000, lr=3e-4, batch_size=4096,
            patience=50):
        """Train each ensemble member on a bootstrapped resample of the data.

        Args:
            obs: np.ndarray [N, obs_dim]
            actions: np.ndarray [N, action_dim]
            patience: stop early if loss doesn't improve for this many epochs
        """
        N = len(obs)
        for k, member in enumerate(self.members):
            idx = np.random.choice(N, size=N, replace=True)
            b_obs = torch.FloatTensor(obs[idx])
            b_actions = torch.FloatTensor(actions[idx])

            optimizer = torch.optim.Adam(member.parameters(), lr=lr)
            dataset = TensorDataset(b_obs, b_actions)
            use_workers = N > 10_000
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=2 if use_workers else 0,
                                pin_memory=(self.device != 'cpu'),
                                persistent_workers=use_workers)

            best_loss = float('inf')
            epochs_no_improve = 0

            member.train()
            pbar = tqdm.trange(num_epochs, desc=f"Ensemble {k+1}/{self.ensemble_size}")
            for epoch in pbar:
                epoch_loss = 0.0
                num_batches = 0
                for obs_batch, act_batch in loader:
                    obs_batch = obs_batch.to(self.device, non_blocking=True)
                    act_batch = act_batch.to(self.device, non_blocking=True)
                    pred = member(obs_batch)
                    loss = nn.MSELoss()(pred, act_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches
                pbar.set_postfix(loss=f'{avg_loss:.6f}')

                if wandb.run is not None and epoch % 10 == 0:
                    wandb.log({
                        f'postbc/ensemble_{k}/loss': avg_loss,
                        'postbc/ensemble_epoch': epoch + k * num_epochs,
                    })

                # Early stopping
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        log(f'Ensemble {k+1}: early stop at epoch {epoch} (best_loss={best_loss:.6f})')
                        break

            log(f'Ensemble member {k + 1}/{self.ensemble_size} trained (loss={avg_loss:.6f})')

    @torch.no_grad()
    def compute_covariance(self, obs):
        """Compute diagonal posterior variance for a batch of observations.

        Args:
            obs: torch.Tensor [N, obs_dim]

        Returns:
            var: torch.Tensor [N, action_dim] diagonal variance
        """
        preds = torch.stack([m(obs) for m in self.members], dim=0)  # [K, N, action_dim]
        mean = preds.mean(dim=0)  # [N, action_dim]
        var = ((preds - mean.unsqueeze(0)) ** 2).mean(dim=0)  # [N, action_dim]
        return var

    def save(self, path):
        """Save all ensemble member state dicts."""
        state = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'ensemble_size': self.ensemble_size,
            'members': [m.state_dict() for m in self.members],
        }
        torch.save(state, path)
        log(f'Saved ensemble to {path}')

    def load(self, path):
        """Load ensemble member state dicts."""
        state = torch.load(path, map_location=self.device)
        for m, sd in zip(self.members, state['members']):
            m.load_state_dict(sd)
        log(f'Loaded ensemble from {path}')


def relabel_actions_postbc(actions, obs, estimator, alpha=1.0):
    """Perturb action targets by posterior-scaled noise.

    Args:
        actions: torch.Tensor [B, action_len, action_dim]
        obs: torch.Tensor [B, obs_dim]
        estimator: fitted PosteriorVarianceEstimator
        alpha: noise scaling factor

    Returns:
        perturbed_actions: torch.Tensor [B, action_len, action_dim]
    """
    with torch.no_grad():
        var = estimator.compute_covariance(obs)  # [B, action_dim]
        std = var.sqrt().unsqueeze(1)  # [B, 1, action_dim]
        noise = torch.randn_like(actions) * std  # [B, action_len, action_dim]
    return actions + alpha * noise
