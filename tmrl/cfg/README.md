# Config Guide

This project uses [Hydra](https://hydra.cc/) with config root at `tmrl/cfg`.

## Entry Configs

- `train.yaml`: offline low-level policy pre-training (flow policies)
- `sac_ogbench.yaml`: online RL on OGBench
- `sac_libero.yaml`: online RL on LIBERO/LIBERO-PRO
- `sac_robot.yaml`: online RL on real robot streams

Each entry config composes:

- `base.yaml` for shared settings (`paths`, `wandb`, optimizer defaults)
- one `model/*` config
- one `dataset/*` config

## Config Groups

- `dataset/`
  - task-specific datasets (`pointmaze-large`, `cube`, `libero_90`, `widowx`, ...)
  - base templates (`base`, `base_ogbench`, `base_libero`, `base_robot`)
- `model/`
  - flow pre-training models (`nfp`, `ngcfp`, `fp`, `gcfp`, `flow`)
  - SAC agent models (`sac_ogbench`, `sac_libero`, `sac_robot`)

## Common Override Patterns

Use Hydra CLI overrides:

```bash
python3 sac_ogbench.py dataset=cube method=tmrl ckpt=/path/to/ckpt
```

```bash
python3 sac_robot.py dataset=droid remote_robot.host=<host> remote_robot.port=<port>
```

```bash
python3 train.py model=ngcfp dataset=cube
```

## Path Defaults

`base.yaml` sets project-relative defaults:

- `paths.data_dir=${paths.project_dir}/datasets`
- `paths.results_dir=${paths.project_dir}/results`
- `paths.wandb_dir=${paths.project_dir}/wandb`
- `paths.ckpt_dir=${paths.project_dir}/checkpoints`

Override any of them at runtime when needed.
