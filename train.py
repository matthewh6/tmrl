import time
from datetime import datetime
from pathlib import Path

import hydra
import torch
from tqdm import tqdm
import yaml
from diffusers.optimization import get_scheduler
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tmrl.utils.data import CubeDataset, MazeDataset, load_data_dict
from tmrl.models.postbc import PosteriorVarianceEstimator, relabel_actions_postbc

import wandb
from tmrl.utils.logging import cprint
from tmrl.utils.common import set_seed


log = lambda msg, color='bright_green': cprint(msg, color)


def train_one_step(
    model: object, optimizer: torch.optim.Optimizer, scheduler: object, batch: dict[str, torch.Tensor]
) -> dict[str, object]:
    loss, info = model(**batch)

    # Step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    info['lr'] = scheduler.get_last_lr()[0]
    return info


@hydra.main(
    version_base=None,
    config_path='tmrl/cfg',
    config_name='train.yaml',
)
def main(cfg: object) -> None:
    if cfg.debug:
        cfg.log_interval = 10

    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            **OmegaConf.to_container(cfg.wandb, resolve=True),
        )

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dict, val_dict = load_data_dict(cfg=cfg)

    # Convert to torch
    train_dict = {k: torch.from_numpy(v) for k, v in train_dict.items()}
    val_dict = {k: torch.from_numpy(v) for k, v in val_dict.items()}

    if 'maze' in cfg.dataset.name:
        train_dataset = MazeDataset(train_dict, cfg.dataset.action_len, cfg.dataset.discount)
        val_dataset = MazeDataset(val_dict, cfg.dataset.action_len, cfg.dataset.discount)
    elif 'cube' in cfg.dataset.name:
        train_dataset = CubeDataset(train_dict, cfg.dataset.action_len, cfg.dataset.discount)
        val_dataset = CubeDataset(val_dict, cfg.dataset.action_len, cfg.dataset.discount)

    # Log batch stats
    log('Sample batch:')
    log('-' * 60)
    sample_batch = train_dataset.sample(cfg.dataset.batch_size)
    for k, v in sample_batch.items():
        log(f'{k:12} | {tuple(v.shape)} | {str(v.dtype):>8} | {v.min():8.4f} | {v.max():8.4f}')
    log('-' * 60)

    # POSTBC: fit or load posterior variance estimator (only for FP models)
    postbc_estimator = None
    if getattr(cfg, 'use_postbc', False):
        postbc_estimator = PosteriorVarianceEstimator(
            obs_dim=cfg.model.obs_dim,
            action_dim=cfg.dataset.action_dim,
            ensemble_size=cfg.postbc_ensemble_size,
            device=device,
        )

        ensemble_cache = Path(cfg.paths.results_dir) / f'postbc_ensemble_{cfg.dataset.name}.pt'
        if ensemble_cache.exists():
            log(f'Loading cached POSTBC ensemble from {ensemble_cache}')
            postbc_estimator.load(ensemble_cache)
        else:
            log(f'Fitting POSTBC ensemble ({cfg.postbc_ensemble_size} members, {cfg.postbc_ensemble_epochs} epochs)...')
            postbc_estimator.fit(
                train_dict['observations'].numpy(),
                train_dict['actions'].numpy(),
                num_epochs=cfg.postbc_ensemble_epochs,
            )
            ensemble_cache.parent.mkdir(parents=True, exist_ok=True)
            postbc_estimator.save(ensemble_cache)

        log(f'POSTBC enabled: alpha={cfg.postbc_alpha}')

    # Create model
    model = instantiate(cfg.model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **cfg.scheduler)
    log('-' * 30 + ' Model Architecture ' + '-' * 30)
    log(model)
    param_count = sum(p.numel() for n, p in model.named_parameters())
    log(f'Total model parameters: {param_count:,}')
    log('-' * 80)

    # Set save path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(cfg.paths.results_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = save_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    first_time = time.time()

    for step in tqdm(range(1, cfg.train_steps + 1), dynamic_ncols=True, desc='Training'):
        log_metrics = {}

        # Train step
        batch = train_dataset.sample(cfg.dataset.batch_size)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # POSTBC: perturb action targets by posterior-scaled noise
        if postbc_estimator is not None:
            if cfg.dataset.name == 'pointmaze-large-navigate-v0':
                obs = batch['obs']
                obs = obs[:, :2]
            else:
                obs = batch['obs']

            batch['actions'] = relabel_actions_postbc(
                batch['actions'], obs, postbc_estimator, alpha=cfg.postbc_alpha,
            )

        info = train_one_step(model, optimizer, scheduler, batch)
        log_metrics.update({f'train/{k}': v for k, v in info.items()})

        # Save checkpoint
        if step % cfg.save_interval == 0:
            ckpt = {
                'model': model.state_dict(),
                'step': step,
            }

            ckpt_path = save_dir / f'model_step_{step}.pt'
            torch.save(ckpt, ckpt_path)

            last_ckpt_path = save_dir / 'last.pt'
            torch.save(ckpt, last_ckpt_path)

            log(f'Saved checkpoint at step {step} to {ckpt_path}')

        # Logging step
        if step % cfg.log_interval == 0:
            model.eval()
            max_key_len = max(len(str(k)) for k in info)
            for k, v in sorted(info.items()):
                log(f'  {k.ljust(max_key_len)} : {v:.4f}')

            val_batch = val_dataset.sample(cfg.dataset.batch_size)
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            with torch.no_grad():
                _, val_info = model(**val_batch)

            log_metrics.update(
                {
                    **{f'val/{k}': v for k, v in val_info.items()},
                    'time/total_time': time.time() - first_time,
                }
            )

        if cfg.use_wandb:
            wandb.log(
                {
                    **log_metrics,
                    'train/global_step': int(step),
                }
            )


if __name__ == '__main__':
    main()
