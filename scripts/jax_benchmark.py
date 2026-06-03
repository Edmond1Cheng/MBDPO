import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "MBDPO"
for path in (str(PKG_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import jax
import jax.numpy as jnp
import numpy as np

from MBDPO.jax_impl import MBDPO
from MBDPO.jax_impl.common import math as jax_math
from MBDPO.jax_impl.parallel import (
    make_data_parallel_train_step,
    max_replica_diff,
    replicate_state,
    shard_batch,
)


def make_cfg(args):
    return SimpleNamespace(
        multitask=False,
        obs="state",
        episodic=False,
        obs_shape={"state": (args.obs_dim,)},
        action_dim=args.action_dim,
        episode_length=500,
        discount_denom=5,
        discount_min=0.95,
        discount_max=0.995,
        horizon=args.horizon,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        enc_dim=args.enc_dim,
        mlp_dim=args.mlp_dim,
        num_enc_layers=2,
        num_q=args.num_q,
        dropout=0.01,
        simnorm_dim=8,
        log_std_min=-10,
        log_std_max=2,
        num_bins=101,
        vmin=-10,
        vmax=10,
        bin_size=20 / 100,
        rho=0.5,
        tau=0.01,
        lr=3e-4,
        enc_lr_scale=0.3,
        grad_clip_norm=20,
        reward_coef=0.1,
        value_coef=0.1,
        consistency_coef=20,
        contrastive_eta=0.01,
        contrastive_coef=1.0,
        contrastive_clip=5.0,
        contrastive_momentum=0.99,
        entropy_coef=1e-4,
        diffusion_steps=args.diffusion_steps,
        diffusion_beta0=1e-4,
        diffusion_betaT=1e-2,
        diffusion_num_samples=args.diffusion_num_samples,
        diffusion_num_elites=args.diffusion_num_elites,
        diffusion_num_pi_trajs=args.diffusion_num_pi_trajs,
        diffusion_temperature=0.5,
        diffusion_action_noise=0.05,
        seed=args.seed,
    )


def make_batch(cfg, seed):
    rng = np.random.default_rng(seed)
    return {
        "obs": rng.normal(
            size=(cfg.horizon + 1, cfg.batch_size, cfg.obs_shape["state"][0])
        ).astype(np.float32),
        "action": rng.uniform(
            -1.0, 1.0, size=(cfg.horizon, cfg.batch_size, cfg.action_dim)
        ).astype(np.float32),
        "reward": rng.normal(size=(cfg.horizon, cfg.batch_size, 1)).astype(np.float32),
        "terminated": np.zeros((cfg.horizon, cfg.batch_size, 1), dtype=np.float32),
    }


def check_math_finite(cfg):
    rng = np.random.default_rng(cfg.seed + 17)
    pred = rng.normal(size=(64, cfg.num_bins)).astype(np.float32)
    target = rng.normal(size=(64, 1)).astype(np.float32)
    jax_loss = np.asarray(
        jax_math.soft_ce(
            jnp.asarray(pred),
            jnp.asarray(target),
            cfg.num_bins,
            cfg.vmin,
            cfg.vmax,
            cfg.bin_size,
        )
    )
    if not np.all(np.isfinite(jax_loss)):
        raise AssertionError("soft_ce produced non-finite values")
    return float(np.max(jax_loss))


def time_call(fn, iters):
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            out,
        )
        times.append(time.perf_counter() - start)
    return {
        "mean_ms": 1000.0 * float(np.mean(times)),
        "p50_ms": 1000.0 * float(np.percentile(times, 50)),
        "p95_ms": 1000.0 * float(np.percentile(times, 95)),
    }


def finite_metrics(metrics):
    values = jax.device_get(metrics)
    return {k: float(np.asarray(v).mean()) for k, v in values.items()}


def benchmark_data_parallel(agent, batch, args):
    available = jax.local_devices()
    if args.data_parallel_devices <= 1 or len(available) < 2:
        return None
    num_devices = min(args.data_parallel_devices, len(available))
    devices = available[:num_devices]
    if batch["obs"].shape[1] % num_devices != 0:
        raise ValueError(
            f"--batch-size={batch['obs'].shape[1]} must be divisible by {num_devices} devices."
        )

    sharded_batch = shard_batch(batch, num_devices)
    dp_state = replicate_state(agent.state, devices)
    dp_step = make_data_parallel_train_step(
        agent.spec, agent.model_tx, agent.pi_tx, devices
    )

    print(f"compile_data_parallel_update_start devices={num_devices}", flush=True)
    compile_start = time.perf_counter()
    dp_state, metrics = dp_step(dp_state, sharded_batch)
    jax.block_until_ready(metrics)
    compile_s = time.perf_counter() - compile_start
    metrics_dict = finite_metrics(metrics)
    bad_metrics = [k for k, v in metrics_dict.items() if not np.isfinite(v)]
    if bad_metrics:
        raise AssertionError(f"non-finite data-parallel metrics: {bad_metrics}")

    for _ in range(args.warmup):
        dp_state, metrics = dp_step(dp_state, sharded_batch)
        jax.block_until_ready(metrics)

    def one_step():
        nonlocal dp_state
        dp_state, metrics = dp_step(dp_state, sharded_batch)
        return metrics

    stats = time_call(one_step, args.iters)
    stats["global_samples_per_s"] = 1000.0 * batch["obs"].shape[1] / stats["mean_ms"]
    replica_max_diff = max_replica_diff(dp_state["params"])
    return {
        "devices": [str(device) for device in devices],
        "compile_s": compile_s,
        "steady": stats,
        "sample_metrics": metrics_dict,
        "replica_max_diff": replica_max_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-dim", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--enc-dim", type=int, default=128)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--diffusion-num-samples", type=int, default=128)
    parser.add_argument("--diffusion-num-elites", type=int, default=32)
    parser.add_argument("--diffusion-num-pi-trajs", type=int, default=8)
    parser.add_argument("--data-parallel-devices", type=int, default=0)
    parser.add_argument("--data-parallel-batch-size", type=int, default=0)
    parser.add_argument("--skip-planner", action="store_true")
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--skip-data-parallel", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = make_cfg(args)
    print("devices:", jax.devices(), flush=True)
    print("math_soft_ce_max:", f"{check_math_finite(cfg):.3e}", flush=True)

    agent = MBDPO(cfg)
    batch = make_batch(cfg, cfg.seed)
    dp_batch = batch
    if args.data_parallel_batch_size > 0:
        dp_cfg = SimpleNamespace(**vars(cfg))
        dp_cfg.batch_size = args.data_parallel_batch_size
        dp_batch = make_batch(dp_cfg, cfg.seed + 101)
    obs = batch["obs"][0, 0]

    plan_compile_s = None
    if not args.skip_planner:
        print("compile_eval_planner_start", flush=True)
        compile_start = time.perf_counter()
        action = agent.act(obs, t0=True, eval_mode=True)
        plan_compile_s = time.perf_counter() - compile_start
        if action.shape != (cfg.action_dim,):
            raise AssertionError(f"bad action shape: {action.shape}")
        if not np.all(np.isfinite(action)) or np.max(np.abs(action)) > 1.0001:
            raise AssertionError("planner returned invalid action")
        print("compile_eval_planner_done", flush=True)

    update_compile_s = None
    metrics = None
    if not args.skip_update:
        print("compile_single_update_start", flush=True)
        compile_start = time.perf_counter()
        metrics = agent.update(batch)
        update_compile_s = time.perf_counter() - compile_start
        bad_metrics = [k for k, v in metrics.items() if not np.isfinite(v)]
        if bad_metrics:
            raise AssertionError(f"non-finite metrics: {bad_metrics}")
        print("compile_single_update_done", flush=True)

    for _ in range(args.warmup):
        if not args.skip_planner:
            agent.act(obs, t0=False, eval_mode=True)
            agent.act(obs, t0=False, eval_mode=False)
        if not args.skip_update:
            agent.update(batch)

    plan_eval_stats = None
    plan_train_stats = None
    if not args.skip_planner:
        print("bench_eval_planner_start", flush=True)
        plan_eval_stats = time_call(
            lambda: agent.act(obs, t0=False, eval_mode=True), args.iters
        )
        print("bench_noisy_planner_start", flush=True)
        plan_train_stats = time_call(
            lambda: agent.act(obs, t0=False, eval_mode=False), args.iters
        )

    update_stats = None
    if not args.skip_update:
        print("bench_single_update_start", flush=True)
        update_stats = time_call(lambda: agent.update(batch), args.iters)
        update_stats["global_samples_per_s"] = (
            1000.0 * cfg.batch_size / update_stats["mean_ms"]
        )

    dp_result = None
    if not args.skip_data_parallel:
        dp_result = benchmark_data_parallel(agent, dp_batch, args)

    if plan_compile_s is not None:
        print("plan_compile_s:", f"{plan_compile_s:.3f}")
    if update_compile_s is not None:
        print("update_compile_s:", f"{update_compile_s:.3f}")
    if plan_eval_stats is not None:
        print("plan_eval_steady:", plan_eval_stats)
    if plan_train_stats is not None:
        print("plan_train_steady:", plan_train_stats)
    if update_stats is not None:
        print("update_steady:", update_stats)
    if dp_result is not None:
        speedup = None
        if update_stats is not None:
            speedup = (
                dp_result["steady"]["global_samples_per_s"]
                / update_stats["global_samples_per_s"]
            )
        print("data_parallel_devices:", dp_result["devices"])
        print("data_parallel_global_batch:", dp_batch["obs"].shape[1])
        print("data_parallel_compile_s:", f"{dp_result['compile_s']:.3f}")
        print("data_parallel_update_steady:", dp_result["steady"])
        print("data_parallel_param_replica_max_diff:", f"{dp_result['replica_max_diff']:.3e}")
        if speedup is not None:
            print("data_parallel_speedup_vs_single:", f"{speedup:.2f}x")
        print(
            "data_parallel_sample_metrics:",
            {k: round(v, 6) for k, v in sorted(dp_result["sample_metrics"].items())},
        )
    if metrics is not None:
        print("sample_metrics:", {k: round(v, 6) for k, v in sorted(metrics.items())})
    print("JAX benchmark completed successfully")


if __name__ == "__main__":
    main()
