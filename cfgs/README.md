# Configuration Guide (`cfgs/`)

This folder contains the Hydra/YAML configuration files used by the training, fine-tuning, evaluation, and parallel-launch scripts.

## File overview

### `config.yaml` (base config for core experiments)

`cfgs/config.yaml` is the **base experiment config**. It defines the common parameter space used by all three core workflows:

1. **Online from scratch** (`scripts/train.py`)
2. **Multi-task offline pretraining** (`scripts/train.py task=mt80|mt30 multitask=true`)
3. **Offline-to-online (O2O) fine-tuning** (`scripts/offline_to_online.py`)

In other words, these workflows share one unified base schema and differ mainly in runtime overrides (e.g., `task`, `steps`, checkpoint paths, etc.).

### `online_parallel_config.yaml` (parallel launcher config)

`cfgs/online_parallel_config.yaml` is a **specialized config** for:

- `scripts/online_parallel_train.py`

It controls how multiple jobs are generated and dispatched in parallel (task set expansion, GPU/worker layout, per-job seed mode, and centralized save behavior).

## Dependency / usage relationship

- `scripts/train.py`, `scripts/evaluate.py`, and `scripts/offline_to_online.py` consume parameters following the base schema in `config.yaml`.
- `scripts/online_parallel_train.py` first reads `online_parallel_config.yaml`, then materializes per-task/per-seed command overrides that are forwarded to the standard training entry (`train_entry`, default `scripts/train.py`).
- Therefore, parallel training = `online_parallel_config.yaml` (dispatch strategy) + `config.yaml`-compatible overrides (actual algorithm/training params).

## Key parameters in `config.yaml`

Below are the most important groups, with emphasis on diffusion-related controls.

### Planning / diffusion parameters (核心)

- `horizon`: rollout horizon used in imagined planning/optimization.
- `diffusion_steps`: number of denoising steps per action-generation pass.
- `diffusion_beta0`, `diffusion_betaT`: start/end noise schedule coefficients.
- `diffusion_num_samples`: number of candidate trajectories/actions sampled from diffusion.
- `diffusion_num_elites`: top-K candidates retained for elite-based improvement.
- `diffusion_num_pi_trajs`: number of policy-conditioned trajectories used during policy update/evaluation.
- `diffusion_temperature`: sampling temperature for controlling exploration/sharpness.
- `diffusion_action_noise`: extra action noise scale (stochasticity regularization).
- `use_score_network`: whether to enable an explicit score network head.
- `score_loss_coef`: weight for score-matching loss when score network is enabled.
- `diffusion_time_embed_dim`: time-step embedding dimension for diffusion-time conditioning.
- `diffusion_eval_compile`, `diffusion_eval_compile_mode`: whether/how to compile diffusion evaluation graph for speed.

### Training stability / objective weighting

- `reward_coef`, `value_coef`, `termination_coef`, `consistency_coef`, `contrastive_eta`, `g_coef`, `rho`: coefficients for different optimization terms.
- `lr`, `enc_lr_scale`, `grad_clip_norm`, `tau`: optimizer and target-update stability controls.
- `discount_denom`, `discount_min`, `discount_max`: discount factor shaping/clipping range.

### Dataset / replay / logging behavior

- `offline_data_mode`: data loading mode for offline training (`in_memory` or `mmap`).
- `save_replay`, `replay_save_dir`, `replay_flush_every_episodes`, `replay_include_terminated`, `replay_task_id`: replay exporting controls.
- `save_reward_csv`, `reward_csv_dir`, `csv_eval_freq`: periodic reward CSV export.
- `save_model_every`, `eval_freq`, `eval_episodes`: model checkpoint and evaluation frequency.

## Key parameters in `online_parallel_config.yaml`

Below are the primary parallelization settings and save-policy controls.

### Parallel scheduling

- `task_set`: task collection alias to expand (e.g., `mt80`, `mt30`).
- `tasks`: optional explicit task list override (empty = infer from `task_set`).
- `gpu_ids`: explicit GPU IDs for dispatch; if empty, uses default indexing logic.
- `num_gpus`: number of GPUs considered by launcher.
- `workers_per_gpu`: concurrent workers per GPU.
- `job_mode`: dispatch mode
- `seeds`: seed list to instantiate repeated runs.

### Per-domain/default step assignment

- `dmc_steps`: default step budget for DMControl tasks.
- `metaworld_steps`: default step budget for MetaWorld tasks.
- `default_steps`: fallback step budget for other domains.

### Compile/evaluation runtime choices

- `compile`, `compile_mode`: model compilation settings for launched jobs.
- `diffusion_eval_compile`, `diffusion_eval_compile_mode`: compilation settings specifically for diffusion evaluation path.
- `eval_freq`, `save_model_every`: periodic evaluation/checkpoint intervals injected into spawned jobs.

### Save/output policies

- `save_replay`, `replay_save_dir`, `replay_flush_every_episodes`, `replay_include_terminated`: replay persistence behavior across parallel workers.
- `save_reward_csv`, `csv_eval_freq`: CSV metric export policy.
- `merge_cleanup_temp`: whether temp merge artifacts are removed after aggregation.
- `common_overrides`: shared key-value overrides appended to every launched command (often used to unify output roots, e.g., `online_save_root`, `reward_csv_dir`).

### Script wiring

- `python_bin`: Python executable used by launcher.
- `train_entry`: downstream training script entry (default `scripts/train.py`).
- `exp_name`, `enable_wandb`, `save_agent`, `save_video`: global experiment metadata and output toggles applied to generated jobs.

## Practical tips

- Keep task-independent algorithm defaults in `config.yaml`, and use CLI overrides for experiment variants.
- Use `online_parallel_config.yaml` only when you need coordinated multi-task/multi-seed parallel dispatch.
- For reproducibility, keep `seeds`, save directories, and `common_overrides` explicit in parallel runs.
