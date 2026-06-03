import functools

import jax
import jax.numpy as jnp
import optax

from .common import math as jm
from .common import world_model as wm
from .mbdpo import (
    _contrastive_loss,
    _global_norm,
    _mc_score_target,
    _percentile_scale,
    _termination_statistics,
    _zeros_like_tree,
)


AXIS_NAME = "mbdpo_devices"


def replicate_state(state, devices):
    """Replicate agent state, keeping a different PRNG key on each device."""
    num_devices = len(devices)

    def replicate_leaf(x):
        return jnp.stack([jnp.asarray(x)] * num_devices, axis=0)

    replicated = jax.tree_util.tree_map(replicate_leaf, state)
    keys = jax.random.split(state["key"], len(devices))
    replicated = dict(replicated)
    replicated["key"] = keys
    return replicated


def shard_batch(batch, num_devices):
    """Shard a T x B batch over devices along the batch dimension."""
    sharded = {}
    for key, value in batch.items():
        dtype = jnp.int32 if key == "task" else jnp.float32
        value = jnp.asarray(value, dtype=dtype)
        if key == "task":
            if value.shape[0] % num_devices != 0:
                raise ValueError(
                    f"Batch dimension for task={value.shape[0]} must divide by {num_devices}."
                )
            per_device = value.shape[0] // num_devices
            sharded[key] = jnp.reshape(value, (num_devices, per_device))
            continue
        if value.shape[1] % num_devices != 0:
            raise ValueError(
                f"Batch dimension for {key}={value.shape[1]} must divide by {num_devices}."
            )
        per_device = value.shape[1] // num_devices
        new_shape = (value.shape[0], num_devices, per_device) + value.shape[2:]
        axes = (1, 0, 2) + tuple(range(3, len(new_shape)))
        sharded[key] = jnp.transpose(jnp.reshape(value, new_shape), axes)
    return sharded


def max_replica_diff(replicated_tree):
    """Return the largest absolute parameter/state difference between replicas."""
    leaves = jax.tree_util.tree_leaves(jax.device_get(replicated_tree))
    max_diff = 0.0
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if arr.shape[:1] == (0,) or arr.shape[0] < 2:
            continue
        diff = float(jnp.max(jnp.abs(arr - arr[:1])))
        max_diff = max(max_diff, diff)
    return max_diff


def make_data_parallel_train_step(spec: wm.ModelSpec, model_tx, pi_tx, devices):
    """Build a pmap data-parallel update step with gradient averaging."""
    rho = jnp.power(spec.rho, jnp.arange(spec.horizon, dtype=jnp.float32))
    pi_rho = jnp.power(spec.rho, jnp.arange(spec.horizon + 1, dtype=jnp.float32))

    @functools.partial(
        jax.pmap,
        axis_name=AXIS_NAME,
        devices=devices,
        donate_argnums=(0,),
    )
    def train_step(state, batch):
        key, k_encode_next, k_encode_z0, k_td, k_q, k_contrast, k_pi_loss, k_score = jax.random.split(
            state["key"], 8
        )
        shared_key, k_pi_q_sync = jax.random.split(state["shared_key"], 2)
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        terminated = batch["terminated"]
        task = batch.get("task", None)

        def model_loss_fn(params):
            next_z = jax.lax.stop_gradient(
                wm.encode(params, obs[1:], spec, task=task, key=k_encode_next)
            )
            target_action, _ = wm.pi(params, next_z, k_td, spec, task=task)
            target_z = wm.task_emb(params, next_z, task, spec)
            target_qs = wm.q_all_from_qs(
                state["target_qs"], target_z, target_action, spec, train=False
            )
            target_q_values = jm.two_hot_inv(
                target_qs, spec.num_bins, spec.vmin, spec.vmax
            )
            target_q = jnp.min(target_q_values[:2], axis=0)
            td_targets = jax.lax.stop_gradient(
                reward + wm.discount(task, spec, reward) * (1.0 - terminated) * target_q
            )

            z0 = wm.encode(params, obs[0], spec, task=task, key=k_encode_z0)

            def rollout_body(z_t, scan_in):
                action_t, next_z_t = scan_in
                z_next = wm.next_latent(params, z_t, action_t, spec, task=task)
                loss_t = jnp.mean(jnp.square(z_next - next_z_t))
                return z_next, (z_next, loss_t)

            _, (pred_zs, consistency_per_t) = jax.lax.scan(
                rollout_body, z0, (action, next_z)
            )
            zs = jnp.concatenate([z0[None], pred_zs], axis=0)
            model_zs = zs[:-1]
            reward_pred = wm.reward(params, model_zs, action, spec, task=task)
            qs = wm.q_all(params, model_zs, action, spec, key=k_q, train=True, task=task)
            termination_pred = (
                wm.termination(params, zs[1:], spec, task=task, unnormalized=True)
                if spec.episodic
                else jnp.zeros_like(terminated)
            )

            consistency_loss = jnp.sum(consistency_per_t * rho) / spec.horizon
            reward_loss = (
                jnp.sum(
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
                / spec.horizon
            )
            value_ce = jm.soft_ce(
                qs,
                td_targets[None],
                spec.num_bins,
                spec.vmin,
                spec.vmax,
                spec.bin_size,
            )
            value_loss = (
                jnp.sum(jnp.mean(value_ce, axis=(2, 3)) * rho[None, :])
                / (spec.horizon * spec.num_q)
            )
            contrastive_loss, contrastive_scores = _contrastive_loss(
                params, model_zs, action, k_contrast, spec, task=task
            )
            termination_loss = (
                jnp.mean(jm.bce_with_logits(termination_pred, terminated))
                if spec.episodic
                else jnp.zeros((), dtype=reward.dtype)
            )
            total_loss = (
                spec.consistency_coef * consistency_loss
                + spec.reward_coef * reward_loss
                + spec.termination_coef * termination_loss
                + spec.value_coef * value_loss
                + spec.contrastive_coef * contrastive_loss
            )
            score_loss = jnp.zeros((), dtype=reward.dtype)
            if spec.use_score_network:
                score_task = jnp.reshape(task, (-1,))[:1] if task is not None else None
                x_tau, tau_idx, target_score = _mc_score_target(
                    params,
                    zs[0, 0:1],
                    score_task,
                    k_score,
                    spec,
                    state["contrastive_mean"],
                    state["contrastive_std"],
                )
                pred_score = wm.score(
                    params,
                    zs[0, 0:1],
                    x_tau[None],
                    tau_idx,
                    spec,
                    task=score_task,
                )[0]
                score_loss = jnp.mean(
                    jnp.square(pred_score - jax.lax.stop_gradient(target_score))
                )
                total_loss = total_loss + spec.score_loss_coef * score_loss
            aux = {
                "zs": jax.lax.stop_gradient(zs),
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "contrastive_loss": contrastive_loss,
                "termination_loss": termination_loss,
                "score_loss": score_loss,
                "contrastive_scores": jax.lax.stop_gradient(contrastive_scores),
                "termination_pred": jax.lax.stop_gradient(termination_pred),
            }
            return total_loss, aux

        (model_loss, aux), model_grads = jax.value_and_grad(
            model_loss_fn, has_aux=True
        )(state["params"])
        model_grads = dict(model_grads)
        model_grads["pi"] = _zeros_like_tree(model_grads["pi"])
        model_grads = jax.lax.pmean(model_grads, AXIS_NAME)
        model_loss = jax.lax.pmean(model_loss, AXIS_NAME)
        model_grad_norm = _global_norm(model_grads)
        model_updates, model_opt_state = model_tx.update(
            model_grads, state["model_opt"], state["params"]
        )
        params = wm.clip_task_embeddings(
            optax.apply_updates(state["params"], model_updates), spec
        )

        scores = aux["contrastive_scores"]
        batch_mean = jax.lax.pmean(jnp.mean(scores), AXIS_NAME)
        batch_sq_mean = jax.lax.pmean(jnp.mean(jnp.square(scores)), AXIS_NAME)
        batch_std = jnp.maximum(
            jnp.sqrt(jnp.maximum(batch_sq_mean - jnp.square(batch_mean), 0.0)), 1e-6
        )
        contrastive_mean = (
            state["contrastive_mean"] * spec.contrastive_momentum
            + batch_mean * (1.0 - spec.contrastive_momentum)
        )
        contrastive_std = (
            state["contrastive_std"] * spec.contrastive_momentum
            + batch_std * (1.0 - spec.contrastive_momentum)
        )

        def pi_loss_fn(pi_params):
            params_for_pi = dict(params)
            params_for_pi["pi"] = pi_params
            pi_action, pi_info = wm.pi(
                params_for_pi, aux["zs"], k_pi_loss, spec, task=task
            )
            qs = wm.q_value(
                params_for_pi,
                aux["zs"],
                pi_action,
                spec,
                "avg",
                key=k_pi_q_sync,
                task=task,
            )
            global_q0 = jax.lax.all_gather(qs[0], AXIS_NAME)
            global_q0 = jnp.reshape(global_q0, (-1,) + qs[0].shape[1:])
            new_scale = state["scale"] + spec.tau * (
                _percentile_scale(global_q0) - state["scale"]
            )
            scaled_qs = qs / new_scale
            loss = -jnp.mean(
                jnp.mean(
                    spec.entropy_coef * pi_info["scaled_entropy"] + scaled_qs,
                    axis=(1, 2),
                )
                * pi_rho
            )
            pi_aux = {
                "pi_entropy": jnp.mean(pi_info["entropy"]),
                "pi_scaled_entropy": jnp.mean(pi_info["scaled_entropy"]),
                "pi_scale": jnp.mean(new_scale),
                "new_scale": new_scale,
            }
            return loss, pi_aux

        (pi_loss, pi_aux), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(
            params["pi"]
        )
        pi_grads = jax.lax.pmean(pi_grads, AXIS_NAME)
        pi_loss = jax.lax.pmean(pi_loss, AXIS_NAME)
        pi_grad_norm = _global_norm(pi_grads)
        pi_updates, pi_opt_state = pi_tx.update(pi_grads, state["pi_opt"], params["pi"])
        params = dict(params)
        params["pi"] = optax.apply_updates(params["pi"], pi_updates)

        target_qs = jax.tree_util.tree_map(
            lambda target, online: target + spec.tau * (online - target),
            state["target_qs"],
            params["qs"],
        )
        new_state = {
            "params": params,
            "target_qs": target_qs,
            "model_opt": model_opt_state,
            "pi_opt": pi_opt_state,
            "key": key,
            "shared_key": shared_key,
            "prev_mean": state["prev_mean"],
            "contrastive_mean": contrastive_mean,
            "contrastive_std": contrastive_std,
            "scale": pi_aux["new_scale"],
        }
        metrics = {
            "consistency_loss": aux["consistency_loss"],
            "reward_loss": aux["reward_loss"],
            "value_loss": aux["value_loss"],
            "contrastive_loss": aux["contrastive_loss"],
            "termination_loss": aux["termination_loss"],
            "score_loss": aux["score_loss"],
            "total_loss": model_loss,
            "grad_norm": model_grad_norm,
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": pi_aux["pi_entropy"],
            "pi_scaled_entropy": pi_aux["pi_scaled_entropy"],
            "pi_scale": pi_aux["pi_scale"],
        }
        if spec.episodic:
            termination_rate, termination_f1 = _termination_statistics(
                jax.nn.sigmoid(aux["termination_pred"][-1]), terminated[-1]
            )
            metrics["termination_rate"] = termination_rate
            metrics["termination_f1"] = termination_f1
        metrics = jax.lax.pmean(metrics, AXIS_NAME)
        return new_state, metrics

    return train_step
