# JAX Implementation Results

This directory indexes validation results for the JAX implementation branch. The implementation now lives in the main package paths, for example `MBDPO/mbdpo.py`, `MBDPO/diffusion.py`, and `MBDPO/common/world_model.py`.

These numbers are not the paper's full benchmark table. They are targeted checks for semantic alignment, speed, and single-environment viability against the original reference implementation.

## Validation Scope

Validated:

- Single-task state online training/evaluation on `cheetah-run`.
- Fixed-replay reference/JAX loss and reward comparisons on the same collected replay.
- Offline fixed-data comparison on an extracted `cheetah-run-back` dataset.
- JIT update and diffusion planner smoke tests.
- Synthetic state/rgb, episodic termination, score-loss, multitask masking, and pmap correctness paths.
- Multi-device gradient averaging via `pmap + pmean` correctness checks.

Not fully validated:

- All 121 paper tasks.
- Full MT30/MT80 paper-scale final quality.
- O2O fine-tuning quality.
- Visual RL closed-loop training beyond synthetic rgb path and code-level parity checks.
- Large multi-node / hundred-GPU runs.

## Key Takeaways

- JAX and reference loss curves are close in scale/trend under fixed replay, but exact equality is not expected because initialization and stochastic sampling are not bitwise shared.
- The closed-loop online comparison is promising but still only a small set of seeds/tasks. It should not be reported as full paper reproduction.
- The strongest speed result is update weak scaling: 4 GPUs at batch 4096 reached `380,783 samples/s`, `3.99x` the single-GPU batch-1024 throughput.
- Fixed-replay online comparison over 10k updates reached similar final reward and loss scales, with JAX about `2.03x` faster per logged update and `4.83x` faster wall-clock in that run.
- Offline 100k same-data training had JAX about `1.55x` faster per logged update and `2.03x` faster wall-clock. Torch had the higher final offline eval reward on this single seed.

## Summary Table

Raw CSV/NPZ files from experiment runs are intentionally ignored for GitHub. The table below is the curated summary; the raw logs and CSVs remain on the remote experiment workspace when needed for deeper inspection.

| Experiment | Scope | Result |
| --- | --- | --- |
| Update weak scaling | Synthetic update benchmark | JAX single GPU batch 1024: `95,462 samples/s`; JAX 4 GPU batch 4096: `380,783 samples/s`; `3.99x` throughput scaling |
| Planner smoke | Tiny compiled planner | Eval planner mean `1.233ms`; noisy planner mean `0.876ms` after compile |
| Online fixed replay 10k | `cheetah-run`, 3 seeds, same replay/sampler | Final eval reward mean JAX `72.563`, Torch `67.435`; final update mean JAX `18.54ms`, Torch `37.69ms`; wall-clock JAX `607.36s`, Torch `2930.48s` |
| Compressed online 100k | `cheetah-run`, seed 81, closed loop | 50k eval matched: JAX `542.309`, Torch `541.7`; 100k eval JAX `766.558`, Torch `708.6` |
| Compressed offline 100k | `cheetah-run-back`, seed 82, same data | Final eval reward JAX `135.839`, Torch `159.704`; final total loss JAX `0.282906`, Torch `0.304769`; update JAX `50.572ms`, Torch `78.225ms` |

## Primary Artifacts

GitHub tracks this summary and the source code. Raw CSV/NPZ files, generated
plots, PDFs, Hydra outputs, and `PROGRESS.md` are intentionally ignored so this
branch stays lightweight.

The local and felis workspaces still contain the generated figures when needed
for inspection:

- `results/compressed_paper/compressed_paper_fullsetting_final.png`
- `results/comparison_big_online/online_seed_3seed_10000_reward_loss_polished.png`
- `results/offline_eval_cheetahback/cheetah_run_back_offline_eval_reward_loss_polished.png`

## Reproduction Notes

The validated remote environment was `felis:/nfs-shared-2/zhancun/workspace/MBDPO` using `/scratch/muzhancun/miniconda3/envs/mbdpo-jax`.

For GPU JAX on felis, `jaxlib 0.7.1` required cuDNN `>=9.8`; the environment was fixed by installing `nvidia-cudnn-cu12==9.8.0.87`.

Use `CUDA_VISIBLE_DEVICES` explicitly and avoid GPUs with active external memory/utilization. The recorded large comparisons used free GPUs only and monitored `nvidia-smi` before launch.
