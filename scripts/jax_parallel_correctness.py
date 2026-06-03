import argparse
import functools
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "MBDPO"
for path in (str(PKG_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from MBDPO.jax_impl.common import world_model as wm
from MBDPO.jax_impl.common import math as jm
from MBDPO.jax_impl.parallel import max_replica_diff, replicate_state, shard_batch

AXIS_NAME = "correctness_devices"


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
        num_q=2,
        dropout=0.0,
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
        diffusion_steps=2,
        diffusion_beta0=1e-4,
        diffusion_betaT=1e-2,
        diffusion_num_samples=4,
        diffusion_num_elites=0,
        diffusion_num_pi_trajs=0,
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
    }


def deterministic_loss(params, batch, spec):
    obs = batch["obs"]
    action = batch["action"]
    reward = batch["reward"]
    rho = jnp.power(spec.rho, jnp.arange(spec.horizon, dtype=jnp.float32))

    next_z = wm.encode(params, obs[1:], spec)
    z0 = wm.encode(params, obs[0], spec)

    def rollout_body(z_t, scan_in):
        action_t, next_z_t = scan_in
        z_next = wm.next_latent(params, z_t, action_t, spec)
        loss_t = jnp.mean(jnp.square(z_next - next_z_t))
        return z_next, (z_next, loss_t)

    _, (pred_zs, consistency_per_t) = jax.lax.scan(
        rollout_body, z0, (action, next_z)
    )
    model_zs = jnp.concatenate([z0[None], pred_zs[:-1]], axis=0)
    reward_pred = wm.reward(params, model_zs, action, spec)
    qs = wm.q_all(params, model_zs, action, spec, train=False)
    f_logits = wm.contrastive_f(params, model_zs, action, spec)
    pi_mean = wm.pi_mean(params, z0, spec)

    reward_loss = jnp.sum(
        jnp.mean(
            jm.soft_ce(
                reward_pred,
                reward,
                spec.num_bins,
                spec.vmin,
                spec.vmax,
                spec.bin_size,
            ),
            axis=(1, 2),
        )
        * rho
    )
    q_target = jnp.zeros_like(reward)
    q_loss = jnp.sum(
        jnp.mean(
            jm.soft_ce(
                qs,
                q_target[None],
                spec.num_bins,
                spec.vmin,
                spec.vmax,
                spec.bin_size,
            ),
            axis=(2, 3),
        )
        * rho[None]
    )
    consistency_loss = jnp.sum(consistency_per_t * rho)
    f_loss = jnp.mean(jm.bce_with_logits(f_logits, jnp.ones_like(f_logits)))
    pi_loss = jnp.mean(jnp.square(pi_mean))
    return (
        spec.consistency_coef * consistency_loss
        + spec.reward_coef * reward_loss
        + spec.value_coef * q_loss
        + spec.contrastive_coef * f_loss
        + 0.01 * pi_loss
    )


def tree_max_abs_diff(a, b):
    max_diff = 0.0
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        max_diff = max(max_diff, float(jnp.max(jnp.abs(x - y))))
    return max_diff


def make_pmap_step(spec, tx, devices):
    @functools.partial(jax.pmap, axis_name=AXIS_NAME, devices=devices)
    def step(state, batch):
        loss, grads = jax.value_and_grad(deterministic_loss)(
            state["params"], batch, spec
        )
        grads = jax.lax.pmean(grads, AXIS_NAME)
        loss = jax.lax.pmean(loss, AXIS_NAME)
        updates, opt_state = tx.update(grads, state["opt"], state["params"])
        params = optax.apply_updates(state["params"], updates)
        return {"params": params, "opt": opt_state}, loss

    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--obs-dim", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--enc-dim", type=int, default=64)
    parser.add_argument("--mlp-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    args = parser.parse_args()

    available = jax.local_devices()
    if len(available) < args.devices:
        raise RuntimeError(f"Need {args.devices} devices, got {len(available)}.")
    devices = available[: args.devices]
    if args.batch_size % args.devices != 0:
        raise ValueError("--batch-size must be divisible by --devices.")

    cfg = make_cfg(args)
    spec = wm.spec_from_cfg(cfg)
    key = jax.random.PRNGKey(args.seed)
    params = wm.init_params(key, spec)
    tx = optax.sgd(spec.lr) if args.optimizer == "sgd" else optax.adam(spec.lr)
    opt_state = tx.init(params)
    batch = make_batch(cfg, args.seed + 13)
    jax_batch = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in batch.items()}

    @jax.jit
    def single_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(deterministic_loss)(params, batch, spec)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("devices:", devices, flush=True)
    print("optimizer:", args.optimizer, flush=True)
    print("compile_single_start", flush=True)
    start = time.perf_counter()
    single_params, _, single_loss = single_step(params, opt_state, jax_batch)
    jax.block_until_ready(single_loss)
    print("compile_single_s:", f"{time.perf_counter() - start:.3f}", flush=True)

    print("compile_pmap_start", flush=True)
    pmap_step = make_pmap_step(spec, tx, devices)
    p_state = replicate_state({"params": params, "opt": opt_state, "key": key}, devices)
    sharded = shard_batch(jax_batch, args.devices)
    start = time.perf_counter()
    p_state, p_loss = pmap_step(p_state, sharded)
    jax.block_until_ready(p_loss)
    print("compile_pmap_s:", f"{time.perf_counter() - start:.3f}", flush=True)

    p_params = jax.tree_util.tree_map(lambda x: x[0], jax.device_get(p_state["params"]))
    param_diff = tree_max_abs_diff(single_params, p_params)
    loss_diff = float(jnp.max(jnp.abs(single_loss - jax.device_get(p_loss)[0])))
    replica_diff = max_replica_diff(p_state["params"])

    print("single_loss:", float(single_loss))
    print("pmap_loss:", float(jax.device_get(p_loss)[0]))
    print("loss_max_abs_diff:", f"{loss_diff:.6e}")
    print("param_max_abs_diff:", f"{param_diff:.6e}")
    print("replica_param_max_abs_diff:", f"{replica_diff:.6e}")
    if param_diff > args.atol or loss_diff > args.atol or replica_diff > args.atol:
        raise AssertionError(
            f"distributed correctness failed: param_diff={param_diff:.3e}, "
            f"loss_diff={loss_diff:.3e}, replica_diff={replica_diff:.3e}"
        )
    print("JAX pmap correctness completed successfully")


if __name__ == "__main__":
    main()
