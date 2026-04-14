# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

**Environment setup:**
```bash
mamba env create -f conda_environment.yaml  # Linux/CUDA
conda env create -f conda_environment_macos.yaml  # macOS dev only
```

**Training:**
```bash
python train.py --config-name=train_diffusion_unet_image_workspace
python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0
```

**Evaluation:**
```bash
python eval.py --checkpoint data/outputs/.../checkpoints/latest.ckpt -o output_dir -d cuda:0
```

**Multi-GPU distributed training (Ray):**
```bash
ray start --head --num-gpus=3
python ray_train_multirun.py --config-name=train_diffusion_unet_lowdim_workspace --seeds=42,43,44 --monitor_key=test/mean_score gpu_ids='[0,1,2]'
```

**Tests** (run individually, no pytest markers):
```bash
python tests/test_replay_buffer.py
python tests/test_robomimic_image_runner.py
```

## Architecture Overview

The codebase implements an imitation learning framework where adding N tasks and M policy methods requires only O(N+M) code via strict separation of concerns:

- **Task** = `dataset` + `env_runner` + `shape_meta` (defined in `config/task/*.yaml`)
- **Method** = `policy` + `workspace` (training loop)

These are composed at runtime by Hydra config; any policy can run on any task.

### Core Execution Path

```
train.py  →  BaseWorkspace.run()  →  training loop
                ├── hydra.utils.instantiate(cfg.policy)
                ├── hydra.utils.instantiate(cfg.task.dataset)
                ├── dataset.get_normalizer() → policy.set_normalizer()
                └── env_runner.run(policy)  →  test/mean_score
```

### Configuration System (Hydra)

All configs live in `diffusion_policy/config/`. Root configs are `train_*.yaml`; they merge a `task/*.yaml` via `defaults:`. Every class is referenced by `_target_:` and instantiated dynamically:

```yaml
policy:
  _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
  shape_meta: ${shape_meta}   # interpolated from task config
```

Custom resolvers: `${eval:'${n_obs_steps}-1'}` executes Python; `${now:%Y.%m.%d}` inserts timestamps.

### Key Module Roles

| Directory | Role |
|-----------|------|
| `workspace/` | Training lifecycle, checkpoint save/load (`BaseWorkspace`) |
| `policy/` | Policy implementations (`predict_action`, `set_normalizer`, loss computation) |
| `dataset/` | Datasets extending `BaseLowdimDataset` / `BaseImageDataset` |
| `env_runner/` | Vectorized simulation evaluators — return `{test/mean_score: ...}` |
| `model/diffusion/` | `ConditionalUnet1D`, `TransformerForDiffusion`, `EMAModel` |
| `model/vision/` | `MultiImageObsEncoder` (ResNet backbone) |
| `common/` | `ReplayBuffer` (zarr storage), `SequenceSampler`, `LinearNormalizer`, `TopKCheckpointManager` |
| `shared_memory/` | Lock-free `SharedMemoryRingBuffer` (FILO) and `SharedMemoryQueue` (FIFO) for real robot |
| `real_world/` | `RealEnv`, `SingleRealsense`, `RTDEInterpolationController` for UR5 |

### Data Pipeline

Demonstrations are stored in zarr format via `ReplayBuffer`:
```
data.zarr/
├── data/action  (N, Da)
├── data/obs_key (N, ...)
└── meta/episode_ends (num_episodes,)
```

`SequenceSampler` slides a window across episodes (with configurable `pad_before`/`pad_after`). Datasets return `{'obs': ..., 'action': (T, Da)}`. Normalization is extracted once via `dataset.get_normalizer()` and injected into the policy.

### Policy Interface

All policies implement:
- `predict_action(obs_dict) → {'action': Tensor}` — inference
- `set_normalizer(normalizer)` — inject `LinearNormalizer`
- `compute_loss(batch)` — training (diffusion policies compute MSE on noise prediction)

Diffusion policies use DDPM/DDIM schedulers from `diffusers`. EMA shadow model is maintained in workspaces via `EMAModel` and swapped in for evaluation.

### Checkpoint Format

Checkpoints store both `state_dicts` and `pickles` alongside the full `OmegaConf` config, enabling exact reproduction. `TopKCheckpointManager` retains the best-k based on a monitored metric (e.g., `test/mean_score`).

### Real Robot

The real-world stack uses multiprocessing with shared memory for zero-copy data transfer. `RealEnv` splits the gym interface into `get_obs()` and `exec_actions()` for asynchronous operation. Camera frames use `SharedMemoryRingBuffer`; robot commands use `SharedMemoryQueue`.
