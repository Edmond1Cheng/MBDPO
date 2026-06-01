# Configuration Guide (`cfgs/`)

This JAX branch uses one Hydra base config:

- `config.yaml`: online single-task training and multi-task offline pretraining.

The training entry is `scripts/train.py`.

## Common Usage

```bash
# Online single-task training
python scripts/train.py task=cheetah-run steps=100000 model_size=1

# Offline multi-task training from NumPy .npz chunks
python scripts/train.py task=mt30 data_dir=/path/to/mt30_npz_chunks steps=100000
```

## Important Groups

- Environment: `task`, `obs`, `episodic`
- Training: `steps`, `batch_size`, `buffer_size`, `seed`
- Objective weights: `reward_coef`, `value_coef`, `termination_coef`, `consistency_coef`, `contrastive_eta`
- Diffusion planner: `horizon`, `diffusion_steps`, `diffusion_num_samples`, `diffusion_num_elites`, `diffusion_num_pi_trajs`, `diffusion_temperature`
- JAX logging/runtime helpers: `jax_log_freq`, `eval_freq`, `eval_episodes`

Raw run CSV/NPZ artifacts are ignored by Git; keep curated result summaries and plots under `results/`.
