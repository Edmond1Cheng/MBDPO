import functools
import pickle
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import math as jm
from .common import world_model as wm
from .diffusion import make_plan_step


def _zeros_like_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _label_tree(tree, label):
    return jax.tree_util.tree_map(lambda _: label, tree)


def make_model_tx(spec: wm.ModelSpec, params):
    labels = {
        "encoder": _label_tree(params["encoder"], "encoder"),
        "dynamics": _label_tree(params["dynamics"], "model"),
        "reward": _label_tree(params["reward"], "model"),
        "pi": _label_tree(params["pi"], "policy"),
        "f": _label_tree(params["f"], "model"),
        "termination": _label_tree(params["termination"], "model"),
        "score": _label_tree(params["score"], "model"),
        "qs": _label_tree(params["qs"], "model"),
    }
    if spec.multitask:
        labels["task_emb"] = _label_tree(params["task_emb"], "model")
    return optax.chain(
        optax.clip_by_global_norm(spec.grad_clip_norm),
        optax.multi_transform(
            {
                "encoder": optax.adam(spec.lr * spec.enc_lr_scale),
                "model": optax.adam(spec.lr),
                "policy": optax.set_to_zero(),
            },
            labels,
        ),
    )


def _global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves]))


def _percentile_scale(x):
    flat = jnp.sort(jnp.reshape(x, (x.shape[0], -1)), axis=0)
    n = flat.shape[0]
    positions = jnp.asarray([5.0, 95.0], dtype=x.dtype) * (n - 1) / 100.0
    floored = jnp.floor(positions).astype(jnp.int32)
    ceiled = jnp.minimum(floored + 1, n - 1)
    w_ceil = positions - floored.astype(x.dtype)
    w_floor = 1.0 - w_ceil
    values = flat[floored] * w_floor[:, None] + flat[ceiled] * w_ceil[:, None]
    return jnp.maximum(jnp.reshape(values[1] - values[0], x.shape[1:]), 1.0)


def _normalize_contrastive(score, mean, std, clip):
    std = jnp.maximum(std, 1e-6)
    return jnp.clip((score - mean) / std, -clip, clip)


def _estimate_value(
    params,
    z,
    actions,
    spec: wm.ModelSpec,
    contrastive_mean,
    contrastive_std,
    key,
    task=None,
):
    key, tail_pi_key, tail_q_key = jax.random.split(key, 3)

    def body(carry, action_t):
        z_t, ret, discount_t, termination_t = carry
        reward_t = jm.two_hot_inv(
            wm.reward(params, z_t, action_t, spec, task=task),
            spec.num_bins,
            spec.vmin,
            spec.vmax,
        )
        f_score = wm.contrastive_f(params, z_t, action_t, spec, task=task)
        f_norm = _normalize_contrastive(
            f_score, contrastive_mean, contrastive_std, spec.contrastive_clip
        )
        shaped_reward = reward_t + spec.contrastive_eta * f_norm
        z_next = wm.next_latent(params, z_t, action_t, spec, task=task)
        ret = ret + discount_t * (1.0 - termination_t) * shaped_reward
        discount_t = discount_t * wm.discount(task, spec, z_t)
        if spec.episodic:
            termination_t = jnp.clip(
                termination_t
                + (wm.termination(params, z_next, spec, task=task) > 0.5).astype(
                    z_t.dtype
                ),
                max=1.0,
            )
        return (z_next, ret, discount_t, termination_t), None

    init = (
        z,
        jnp.zeros((z.shape[0], 1), dtype=z.dtype),
        jnp.ones((z.shape[0], 1), dtype=z.dtype),
        jnp.zeros((z.shape[0], 1), dtype=z.dtype),
    )
    (z_final, ret, discount_t, termination_t), _ = jax.lax.scan(body, init, actions)
    tail_action, _ = wm.pi(params, z_final, tail_pi_key, spec, task=task)
    return ret + discount_t * (1.0 - termination_t) * wm.q_value(
        params, z_final, tail_action, spec, "avg", key=tail_q_key, task=task
    )


def _task_for_horizon(task, horizon):
    if task is None:
        return None
    return jnp.repeat(jnp.reshape(task, (-1,)), horizon)


def _contrastive_loss(params, zs, actions, key, spec: wm.ModelSpec, task=None):
    z_flat = jax.lax.stop_gradient(jnp.reshape(zs, (-1, zs.shape[-1])))
    a_pos = jnp.reshape(jnp.swapaxes(actions, 0, 1), (-1, actions.shape[-1]))
    task_flat = _task_for_horizon(task, spec.horizon)
    pos_logits = wm.contrastive_f(params, z_flat, a_pos, spec, task=task_flat)
    pos_loss = jm.bce_with_logits(pos_logits, jnp.ones_like(pos_logits))

    k_pi, k_noise, k_rand = jax.random.split(key, 3)
    pi_action, _ = wm.pi(params, z_flat, k_pi, spec, task=task_flat)
    noisy_action = jnp.clip(
        pi_action
        + spec.contrastive_neg_noise * jax.random.normal(k_noise, pi_action.shape),
        -1.0,
        1.0,
    )
    random_action = jax.random.uniform(k_rand, pi_action.shape, minval=-1.0, maxval=1.0)
    candidates = jnp.stack(
        [
            jax.lax.stop_gradient(pi_action),
            jax.lax.stop_gradient(noisy_action),
            random_action,
        ],
        axis=1,
    )
    z_candidates = jnp.repeat(z_flat[:, None, :], candidates.shape[1], axis=1)
    candidate_logits = wm.contrastive_f(
        params,
        jnp.reshape(z_candidates, (-1, z_flat.shape[-1])),
        jnp.reshape(candidates, (-1, candidates.shape[-1])),
        spec,
        task=(
            jnp.repeat(task_flat, candidates.shape[1])
            if task_flat is not None
            else None
        ),
    )
    candidate_logits = jnp.reshape(candidate_logits, (-1, candidates.shape[1], 1))
    hard_idx = jnp.argmax(jax.lax.stop_gradient(candidate_logits[..., 0]), axis=1)
    hard_neg = candidates[jnp.arange(candidates.shape[0]), hard_idx]
    neg_logits = wm.contrastive_f(params, z_flat, hard_neg, spec, task=task_flat)
    neg_loss = jm.bce_with_logits(neg_logits, jnp.zeros_like(neg_logits))
    loss = 0.5 * (jnp.mean(pos_loss) + jnp.mean(neg_loss))
    scores = jnp.concatenate([pos_logits, neg_logits], axis=0)
    return loss, scores


def _mc_score_target(
    params,
    z0,
    task,
    key,
    spec: wm.ModelSpec,
    contrastive_mean,
    contrastive_std,
):
    num_samples = int(spec.diffusion_num_samples_mf)
    num_steps = max(spec.diffusion_steps, 2)
    k_tau, k_x, k_samples, k_value = jax.random.split(key, 4)
    betas = jnp.linspace(spec.diffusion_beta0, spec.diffusion_betaT, num_steps)
    alphas = 1.0 - betas
    alpha_bar = jnp.cumprod(alphas)
    tau = jax.random.randint(k_tau, (1,), 1, num_steps)
    alpha_bar_tau = alpha_bar[tau[0]]
    x_tau = jax.random.normal(k_x, (spec.horizon, spec.action_dim), dtype=z0.dtype)
    mean_cond = x_tau / jnp.sqrt(alpha_bar_tau)
    std_cond = jnp.sqrt((1.0 - alpha_bar_tau) / alpha_bar_tau)
    a0_samples = jnp.clip(
        mean_cond[None]
        + std_cond
        * jax.random.normal(
            k_samples, (num_samples, spec.horizon, spec.action_dim), dtype=z0.dtype
        ),
        -1.0,
        1.0,
    )
    values = _estimate_value(
        params,
        jnp.repeat(z0, num_samples, axis=0),
        jnp.swapaxes(a0_samples, 0, 1),
        spec,
        contrastive_mean,
        contrastive_std,
        k_value,
        task=task,
    )
    values = jnp.reshape(jnp.nan_to_num(values, nan=0.0), (num_samples, -1)).mean(
        axis=-1
    )
    logits = (values - jnp.mean(values)) / (jnp.std(values, ddof=1) + 1e-6)
    weights = jax.nn.softmax(logits / max(spec.diffusion_temperature, 1e-6), axis=0)
    a_bar = jnp.sum(weights[:, None, None] * a0_samples, axis=0)
    target_score = (-x_tau + jnp.sqrt(alpha_bar_tau) * a_bar) / (
        1.0 - alpha_bar_tau + 1e-8
    )
    return x_tau, tau, target_score


def _termination_statistics(pred, target):
    pred = jnp.squeeze(pred, axis=-1)
    target = jnp.squeeze(target, axis=-1)
    rate = jnp.mean(target)
    tp = jnp.sum((pred > 0.5) & (target == 1.0))
    fn = jnp.sum((pred <= 0.5) & (target == 1.0))
    fp = jnp.sum((pred > 0.5) & (target == 0.0))
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = 2.0 * (precision * recall) / (precision + recall + 1e-9)
    return rate, f1


def _make_train_step(spec: wm.ModelSpec, model_tx, pi_tx):
    rho = jnp.power(spec.rho, jnp.arange(spec.horizon, dtype=jnp.float32))
    pi_rho = jnp.power(spec.rho, jnp.arange(spec.horizon + 1, dtype=jnp.float32))

    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch):
        key, k_encode_next, k_encode_z0, k_td, k_q, k_contrast, k_pi_loss, k_score = jax.random.split(
            state["key"], 8
        )
        shared_key, k_pi_q = jax.random.split(state["shared_key"], 2)
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
                score_task = (
                    jnp.reshape(task, (-1,))[:1] if task is not None else None
                )
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
                score_loss = jnp.mean(jnp.square(pred_score - jax.lax.stop_gradient(target_score)))
                total_loss = total_loss + spec.score_loss_coef * score_loss
            aux = {
                "zs": jax.lax.stop_gradient(zs),
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "contrastive_loss": contrastive_loss,
                "termination_loss": termination_loss,
                "score_loss": score_loss,
                "total_loss": total_loss,
                "contrastive_scores": jax.lax.stop_gradient(contrastive_scores),
                "termination_pred": jax.lax.stop_gradient(termination_pred),
            }
            return total_loss, aux

        (model_loss, aux), model_grads = jax.value_and_grad(
            model_loss_fn, has_aux=True
        )(state["params"])
        model_grads = dict(model_grads)
        model_grads["pi"] = _zeros_like_tree(model_grads["pi"])
        model_grad_norm = _global_norm(model_grads)
        model_updates, model_opt_state = model_tx.update(
            model_grads, state["model_opt"], state["params"]
        )
        params = wm.clip_task_embeddings(
            optax.apply_updates(state["params"], model_updates), spec
        )

        scores = aux["contrastive_scores"]
        batch_mean = jnp.mean(scores)
        batch_std = jnp.maximum(jnp.std(scores), 1e-6)
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
                params_for_pi, aux["zs"], pi_action, spec, "avg", key=k_pi_q, task=task
            )
            new_scale = state["scale"] + spec.tau * (_percentile_scale(qs[0]) - state["scale"])
            scaled_qs = qs / new_scale
            loss = -jnp.mean(
                jnp.mean(spec.entropy_coef * pi_info["scaled_entropy"] + scaled_qs, axis=(1, 2))
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
        return new_state, metrics

    return train_step


class MBDPO:
    """JAX MBDPO implementation for state/rgb observations."""

    def __init__(self, cfg):
        if cfg.obs not in {"state", "rgb"}:
            raise ValueError("JAX implementation supports state and rgb observations only.")

        self.cfg = cfg
        self.spec = wm.spec_from_cfg(cfg)
        key = jax.random.PRNGKey(int(cfg.seed))
        key, init_key = jax.random.split(key)
        key, shared_key = jax.random.split(key)
        params = wm.init_params(init_key, self.spec)
        self.model_tx = make_model_tx(self.spec, params)
        self.pi_tx = optax.chain(
            optax.clip_by_global_norm(self.spec.grad_clip_norm),
            optax.adam(self.spec.lr, eps=1e-5),
        )
        self.state = {
            "params": params,
            "target_qs": jax.tree_util.tree_map(lambda x: x.copy(), params["qs"]),
            "model_opt": self.model_tx.init(params),
            "pi_opt": self.pi_tx.init(params["pi"]),
            "key": key,
            "shared_key": shared_key,
            "prev_mean": jnp.zeros(
                (self.spec.horizon, self.spec.action_dim), dtype=jnp.float32
            ),
            "contrastive_mean": jnp.zeros((1,), dtype=jnp.float32),
            "contrastive_std": jnp.ones((1,), dtype=jnp.float32),
            "scale": jnp.ones((1,), dtype=jnp.float32),
        }
        self._train_step = _make_train_step(self.spec, self.model_tx, self.pi_tx)
        self._plan_step = make_plan_step(self.spec, _estimate_value)

    def act(self, obs, t0=False, eval_mode=False, task=None):
        obs = np.asarray(obs, dtype=np.float32)
        task_value = 0 if task is None else int(np.asarray(task).reshape(-1)[0])
        action, prev_mean, key = self._plan_step(
            self.state["params"],
            self.state["prev_mean"],
            self.state["key"],
            jnp.asarray(obs),
            jnp.asarray(bool(t0)),
            jnp.asarray(bool(eval_mode)),
            self.state["contrastive_mean"],
            self.state["contrastive_std"],
            jnp.asarray(task_value, dtype=jnp.int32),
        )
        self.state["prev_mean"] = prev_mean
        self.state["key"] = key
        return np.asarray(jax.device_get(action), dtype=np.float32)

    def update(self, batch: Dict[str, np.ndarray]):
        jax_batch = {}
        for k, v in batch.items():
            dtype = jnp.int32 if k == "task" else jnp.float32
            jax_batch[k] = jnp.asarray(v, dtype=dtype)
        self.state, metrics = self._train_step(self.state, jax_batch)
        return {k: float(np.asarray(jax.device_get(v))) for k, v in metrics.items()}

    def save(self, fp):
        payload = {
            "state": jax.device_get(self.state),
            "spec": self.spec,
        }
        with open(fp, "wb") as f:
            pickle.dump(payload, f)

    def load(self, fp):
        with open(fp, "rb") as f:
            payload = pickle.load(f)
        self.state = payload["state"]
        return self
