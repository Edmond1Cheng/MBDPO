from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import math as jm


ArrayTree = Any


@dataclass(frozen=True)
class ModelSpec:
    obs: str
    obs_shape: tuple[int, ...]
    obs_dim: int
    action_dim: int
    horizon: int
    batch_size: int
    latent_dim: int
    enc_dim: int
    mlp_dim: int
    num_enc_layers: int
    num_q: int
    dropout: float
    simnorm_dim: int
    log_std_min: float
    log_std_max: float
    num_bins: int
    vmin: float
    vmax: float
    bin_size: float
    rho: float
    tau: float
    lr: float
    enc_lr_scale: float
    grad_clip_norm: float
    reward_coef: float
    value_coef: float
    consistency_coef: float
    contrastive_eta: float
    contrastive_coef: float
    contrastive_clip: float
    contrastive_momentum: float
    contrastive_neg_noise: float
    entropy_coef: float
    discount: float
    diffusion_steps: int
    diffusion_beta0: float
    diffusion_betaT: float
    diffusion_num_samples: int
    diffusion_num_samples_mf: int
    diffusion_num_elites: int
    diffusion_num_pi_trajs: int
    diffusion_temperature: float
    diffusion_action_noise: float
    num_channels: int
    shift_aug_pad: int
    multitask: bool
    episodic: bool
    use_score_network: bool
    score_only_inference: bool
    score_loss_coef: float
    termination_coef: float
    task_dim: int
    num_tasks: int
    action_dims: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    discounts: tuple[float, ...]
    diffusion_time_embed_dim: int


def spec_from_cfg(cfg) -> ModelSpec:
    if cfg.obs not in {"state", "rgb"}:
        raise ValueError("JAX implementation supports state and rgb observations only.")
    if cfg.obs == "rgb" and bool(getattr(cfg, "multitask", False)):
        raise ValueError("JAX rgb observations currently support single-task training only; rgb+multitask is unsupported.")
    obs_shape = tuple(int(x) for x in cfg.obs_shape[cfg.obs])
    if cfg.obs == "state":
        obs_dim = int(obs_shape[0])
    else:
        if len(obs_shape) != 3 or obs_shape[-1] != 64:
            raise ValueError(f"JAX rgb observations expect CHW 64x64 input, got {obs_shape}.")
        conv_dim = int(getattr(cfg, "num_channels", 32)) * 4 * 4
        if conv_dim != int(cfg.latent_dim):
            raise ValueError(
                "JAX rgb encoder requires "
                f"num_channels*4*4 == latent_dim; got {conv_dim} vs {cfg.latent_dim}."
            )
        obs_dim = conv_dim
    action_dim = int(cfg.action_dim)
    episode_length = int(cfg.episode_length)
    frac = episode_length / float(cfg.discount_denom)
    discount = min(max((frac - 1.0) / frac, cfg.discount_min), cfg.discount_max)
    tasks = tuple(getattr(cfg, "tasks", [getattr(cfg, "task", "task")]))
    num_tasks = len(tasks) if bool(getattr(cfg, "multitask", False)) else 1
    cfg_action_dims = getattr(cfg, "action_dims", None)
    if not isinstance(cfg_action_dims, (list, tuple)) or len(cfg_action_dims) != num_tasks:
        cfg_action_dims = [action_dim] * num_tasks
    action_dims = tuple(int(x) for x in cfg_action_dims)
    if len(action_dims) != num_tasks:
        action_dims = tuple([action_dim] * num_tasks)
    cfg_episode_lengths = getattr(cfg, "episode_lengths", None)
    if not isinstance(cfg_episode_lengths, (list, tuple)) or len(cfg_episode_lengths) != num_tasks:
        cfg_episode_lengths = [episode_length] * num_tasks
    episode_lengths = tuple(int(x) for x in cfg_episode_lengths)
    if len(episode_lengths) != num_tasks:
        episode_lengths = tuple([episode_length] * num_tasks)
    discounts = tuple(
        float(
            min(
                max(
                    ((ep_len / float(cfg.discount_denom)) - 1.0)
                    / (ep_len / float(cfg.discount_denom)),
                    cfg.discount_min,
                ),
                cfg.discount_max,
            )
        )
        for ep_len in episode_lengths
    )
    return ModelSpec(
        obs=str(cfg.obs),
        obs_shape=obs_shape,
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=int(cfg.horizon),
        batch_size=int(cfg.batch_size),
        latent_dim=int(cfg.latent_dim),
        enc_dim=int(cfg.enc_dim),
        mlp_dim=int(cfg.mlp_dim),
        num_enc_layers=int(cfg.num_enc_layers),
        num_q=int(cfg.num_q),
        dropout=float(getattr(cfg, "dropout", 0.0)),
        simnorm_dim=int(cfg.simnorm_dim),
        log_std_min=float(cfg.log_std_min),
        log_std_max=float(cfg.log_std_max),
        num_bins=int(cfg.num_bins),
        vmin=float(cfg.vmin),
        vmax=float(cfg.vmax),
        bin_size=float(cfg.bin_size),
        rho=float(cfg.rho),
        tau=float(cfg.tau),
        lr=float(cfg.lr),
        enc_lr_scale=float(cfg.enc_lr_scale),
        grad_clip_norm=float(cfg.grad_clip_norm),
        reward_coef=float(cfg.reward_coef),
        value_coef=float(cfg.value_coef),
        consistency_coef=float(cfg.consistency_coef),
        contrastive_eta=float(getattr(cfg, "contrastive_eta", 0.01)),
        contrastive_coef=float(getattr(cfg, "contrastive_coef", 1.0)),
        contrastive_clip=float(getattr(cfg, "contrastive_clip", 5.0)),
        contrastive_momentum=float(getattr(cfg, "contrastive_momentum", 0.99)),
        contrastive_neg_noise=float(getattr(cfg, "contrastive_neg_noise", 0.5)),
        entropy_coef=float(cfg.entropy_coef),
        discount=float(discount),
        diffusion_steps=max(int(cfg.diffusion_steps), 2),
        diffusion_beta0=float(cfg.diffusion_beta0),
        diffusion_betaT=float(cfg.diffusion_betaT),
        diffusion_num_samples=int(cfg.diffusion_num_samples),
        diffusion_num_samples_mf=int(getattr(cfg, "diffusion_num_samples_mf", 64)),
        diffusion_num_elites=int(getattr(cfg, "diffusion_num_elites", 0) or 0),
        diffusion_num_pi_trajs=int(getattr(cfg, "diffusion_num_pi_trajs", 0) or 0),
        diffusion_temperature=float(cfg.diffusion_temperature),
        diffusion_action_noise=float(cfg.diffusion_action_noise),
        num_channels=int(getattr(cfg, "num_channels", 32)),
        shift_aug_pad=int(getattr(cfg, "shift_aug_pad", 3)),
        multitask=bool(getattr(cfg, "multitask", False)),
        episodic=bool(getattr(cfg, "episodic", False)),
        use_score_network=bool(getattr(cfg, "use_score_network", False)),
        score_only_inference=bool(getattr(cfg, "score_only_inference", False)),
        score_loss_coef=float(getattr(cfg, "score_loss_coef", 1.0)),
        termination_coef=float(getattr(cfg, "termination_coef", 1.0)),
        task_dim=int(getattr(cfg, "task_dim", 0) or 0),
        num_tasks=num_tasks,
        action_dims=action_dims,
        episode_lengths=episode_lengths,
        discounts=discounts,
        diffusion_time_embed_dim=int(getattr(cfg, "diffusion_time_embed_dim", 64)),
    )


def _trunc_normal(key, shape, std=0.02):
    # Keep a narrow truncated normal initialization for parity with the reference architecture.
    return jnp.clip(jax.random.normal(key, shape, dtype=jnp.float32) * std, -2.0, 2.0)


def _init_linear(key, in_dim, out_dim, normed=False, zero_weight=False):
    w = (
        jnp.zeros((in_dim, out_dim), dtype=jnp.float32)
        if zero_weight
        else _trunc_normal(key, (in_dim, out_dim))
    )
    layer = {
        "w": w,
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }
    if normed:
        layer["ln_scale"] = jnp.ones((out_dim,), dtype=jnp.float32)
        layer["ln_bias"] = jnp.zeros((out_dim,), dtype=jnp.float32)
    return layer


def _init_conv(key, in_channels, out_channels, kernel_size):
    k_w, k_b = jax.random.split(key)
    fan_in = int(in_channels) * int(kernel_size) * int(kernel_size)
    bound = 1.0 / np.sqrt(fan_in)
    return {
        "w": jax.random.uniform(
            k_w,
            (int(out_channels), int(in_channels), int(kernel_size), int(kernel_size)),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        ),
        "b": jax.random.uniform(
            k_b,
            (int(out_channels),),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        ),
    }


def init_conv_encoder(key, spec: ModelSpec):
    channels = [spec.obs_shape[0], spec.num_channels, spec.num_channels, spec.num_channels, spec.num_channels]
    kernels = [7, 5, 3, 3]
    keys = jax.random.split(key, len(kernels))
    return [
        _init_conv(keys[i], channels[i], channels[i + 1], kernels[i])
        for i in range(len(kernels))
    ]


def init_mlp(key, in_dim, hidden_dims, out_dim, final_norm=False, zero_last=False):
    dims = [int(in_dim)] + [int(d) for d in hidden_dims] + [int(out_dim)]
    keys = jax.random.split(key, len(dims) - 1)
    layers = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = i == len(dims) - 2
        layers.append(
            _init_linear(
                keys[i],
                din,
                dout,
                normed=(not is_last) or final_norm,
                zero_weight=is_last and zero_last,
            )
        )
    return layers


def clip_task_embeddings(params, spec: ModelSpec):
    if not spec.multitask or "task_emb" not in params:
        return params
    emb = params["task_emb"]
    norm = jnp.linalg.norm(emb, axis=-1, keepdims=True)
    params = dict(params)
    params["task_emb"] = emb / jnp.maximum(norm, 1.0)
    return params


def init_params(key, spec: ModelSpec) -> ArrayTree:
    keys = jax.random.split(key, 9 + spec.num_q)
    encoder_hidden = max(spec.num_enc_layers - 1, 1) * [spec.enc_dim]
    common_hidden = 2 * [spec.mlp_dim]
    q_keys = keys[9:]
    encoder = (
        init_mlp(
            keys[0],
            spec.obs_dim + spec.task_dim,
            encoder_hidden,
            spec.latent_dim,
            final_norm=True,
        )
        if spec.obs == "state"
        else init_conv_encoder(keys[0], spec)
    )
    params = {
        "encoder": encoder,
        "dynamics": init_mlp(
            keys[1],
            spec.latent_dim + spec.action_dim + spec.task_dim,
            common_hidden,
            spec.latent_dim,
            final_norm=True,
        ),
        "reward": init_mlp(
            keys[2],
            spec.latent_dim + spec.action_dim + spec.task_dim,
            common_hidden,
            max(spec.num_bins, 1),
            zero_last=True,
        ),
        "pi": init_mlp(
            keys[3],
            spec.latent_dim + spec.task_dim,
            common_hidden,
            2 * spec.action_dim,
        ),
        "f": init_mlp(
            keys[4],
            spec.latent_dim + spec.action_dim + spec.task_dim,
            common_hidden,
            1,
            zero_last=True,
        ),
        "termination": init_mlp(
            keys[5],
            spec.latent_dim + spec.task_dim,
            common_hidden,
            1,
        ),
        "score": init_mlp(
            keys[6],
            spec.latent_dim
            + spec.task_dim
            + spec.horizon * spec.action_dim
            + spec.diffusion_time_embed_dim,
            common_hidden,
            spec.horizon * spec.action_dim,
        ),
        "qs": [
            init_mlp(
                q_keys[i],
                spec.latent_dim + spec.action_dim + spec.task_dim,
                common_hidden,
                max(spec.num_bins, 1),
                zero_last=True,
            )
            for i in range(spec.num_q)
        ],
    }
    if spec.multitask:
        params["task_emb"] = jax.random.uniform(
            keys[7],
            (spec.num_tasks, spec.task_dim),
            minval=-0.02,
            maxval=0.02,
            dtype=jnp.float32,
        )
    return clip_task_embeddings(params, spec)


def _layer_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * scale + bias


def simnorm(x, dim):
    shape = x.shape
    x = jnp.reshape(x, shape[:-1] + (-1, dim))
    x = jax.nn.softmax(x, axis=-1)
    return jnp.reshape(x, shape)


def apply_mlp(
    params,
    x,
    spec: ModelSpec,
    *,
    final_norm=False,
    dropout=0.0,
    key=None,
    train=False,
):
    dropout_keys = None
    if train and dropout > 0.0 and key is not None:
        dropout_keys = jax.random.split(key, max(len(params) - 1, 1))
    for i, layer in enumerate(params):
        is_last = i == len(params) - 1
        x = x @ layer["w"] + layer["b"]
        if is_last and not final_norm:
            return x
        if train and dropout > 0.0 and i == 0 and dropout_keys is not None:
            keep = 1.0 - dropout
            mask = jax.random.bernoulli(dropout_keys[i], keep, x.shape)
            x = jnp.where(mask, x / keep, 0.0)
        x = _layer_norm(x, layer["ln_scale"], layer["ln_bias"])
        x = simnorm(x, spec.simnorm_dim) if is_last and final_norm else jm.mish(x)
    return x


def _shift_aug(x, key, pad):
    if pad <= 0:
        return x.astype(jnp.float32)
    flat_shape = (-1,) + x.shape[-3:]
    x_flat = jnp.reshape(x.astype(jnp.float32), flat_shape)
    _, channels, height, width = x_flat.shape
    x_pad = jnp.pad(
        x_flat,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="edge",
    )
    if key is None:
        shifts = jnp.zeros((x_flat.shape[0], 2), dtype=jnp.int32)
    else:
        shifts = jax.random.randint(
            key,
            (x_flat.shape[0], 2),
            minval=0,
            maxval=2 * pad + 1,
            dtype=jnp.int32,
        )

    def crop(img, shift):
        return jax.lax.dynamic_slice(
            img,
            (0, shift[0], shift[1]),
            (channels, height, width),
        )

    cropped = jax.vmap(crop)(x_pad, shifts)
    return jnp.reshape(cropped, x.shape)


def _conv2d(x, layer, stride):
    y = jax.lax.conv_general_dilated(
        x,
        layer["w"],
        window_strides=(stride, stride),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + layer["b"][None, :, None, None]


def apply_conv_encoder(params, obs, spec: ModelSpec, key=None):
    leading_shape = obs.shape[:-3]
    x = _shift_aug(obs, key, spec.shift_aug_pad)
    x = x / 255.0 - 0.5
    x = jnp.reshape(x, (-1,) + spec.obs_shape)
    strides = [2, 2, 2, 1]
    for i, (layer, stride) in enumerate(zip(params, strides)):
        x = _conv2d(x, layer, stride)
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    x = jnp.reshape(x, leading_shape + (-1,))
    return simnorm(x, spec.simnorm_dim)


def _task_emb(params, task, spec: ModelSpec):
    task = jnp.asarray(task, dtype=jnp.int32)
    return params["task_emb"][task]


def task_emb(params, x, task, spec: ModelSpec):
    if not spec.multitask:
        return x
    emb = _task_emb(params, task, spec)
    target_shape = x.shape[:-1] + (spec.task_dim,)
    if emb.ndim == 1:
        emb = jnp.broadcast_to(emb, target_shape)
    elif emb.ndim + 1 == x.ndim:
        emb = jnp.broadcast_to(emb[None], target_shape)
    else:
        emb = jnp.broadcast_to(emb, target_shape)
    return jnp.concatenate([x, emb], axis=-1)


def action_masks(spec: ModelSpec):
    masks = np.zeros((spec.num_tasks, spec.action_dim), dtype=np.float32)
    for i, dim in enumerate(spec.action_dims):
        masks[i, : int(dim)] = 1.0
    return jnp.asarray(masks)


def action_mask(task, spec: ModelSpec, like=None):
    if not spec.multitask:
        mask = jnp.ones((spec.action_dim,), dtype=jnp.float32)
    else:
        mask = action_masks(spec)[jnp.asarray(task, dtype=jnp.int32)]
    if like is None:
        return mask
    target_shape = like.shape[:-1] + (spec.action_dim,)
    if mask.ndim == 1:
        return jnp.broadcast_to(mask, target_shape)
    if mask.ndim + 1 == like.ndim:
        return jnp.broadcast_to(mask[None], target_shape)
    return jnp.broadcast_to(mask, target_shape)


def action_size(task, spec: ModelSpec, like=None):
    if not spec.multitask:
        size = jnp.asarray(spec.action_dim, dtype=jnp.float32)
    else:
        size = jnp.asarray(spec.action_dims, dtype=jnp.float32)[
            jnp.asarray(task, dtype=jnp.int32)
        ]
    if like is None:
        return size
    target_shape = like.shape[:-1] + (1,)
    size = jnp.reshape(size, size.shape + (1,)) if size.ndim > 0 else size
    if size.ndim == 0:
        return jnp.broadcast_to(size, target_shape)
    if size.ndim + 1 == like.ndim:
        return jnp.broadcast_to(size[None], target_shape)
    return jnp.broadcast_to(size, target_shape)


def discount(task, spec: ModelSpec, like=None):
    if not spec.multitask:
        value = jnp.asarray(spec.discount, dtype=jnp.float32)
    else:
        value = jnp.asarray(spec.discounts, dtype=jnp.float32)[
            jnp.asarray(task, dtype=jnp.int32)
        ]
    if like is None:
        return value
    target_shape = like.shape[:-1] + (1,)
    value = jnp.reshape(value, value.shape + (1,)) if value.ndim > 0 else value
    if value.ndim == 0:
        return jnp.broadcast_to(value, target_shape)
    if value.ndim + 1 == like.ndim:
        return jnp.broadcast_to(value[None], target_shape)
    return jnp.broadcast_to(value, target_shape)


def encode(params, obs, spec: ModelSpec, task=None, key=None):
    if spec.obs == "rgb":
        return apply_conv_encoder(params["encoder"], obs, spec, key=key)
    return apply_mlp(
        params["encoder"],
        task_emb(params, obs, task, spec),
        spec,
        final_norm=True,
    )


def next_latent(params, z, action, spec: ModelSpec, task=None):
    z = task_emb(params, z, task, spec)
    return apply_mlp(
        params["dynamics"],
        jnp.concatenate([z, action], axis=-1),
        spec,
        final_norm=True,
    )


def reward(params, z, action, spec: ModelSpec, task=None):
    z = task_emb(params, z, task, spec)
    return apply_mlp(
        params["reward"], jnp.concatenate([z, action], axis=-1), spec
    )


def termination(params, z, spec: ModelSpec, task=None, unnormalized=False):
    z = task_emb(params, z, task, spec)
    logits = apply_mlp(params["termination"], z, spec)
    return logits if unnormalized else jax.nn.sigmoid(logits)


def contrastive_f(params, z, action, spec: ModelSpec, task=None):
    z = task_emb(params, z, task, spec)
    return apply_mlp(params["f"], jnp.concatenate([z, action], axis=-1), spec)


def _time_embed(tau, dim):
    half = dim // 2
    freq = jnp.exp(
        jnp.arange(half, dtype=jnp.float32)
        * (-jnp.log(10000.0) / max(half - 1, 1))
    )
    angles = tau.astype(jnp.float32)[..., None] * freq[None]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if emb.shape[-1] < dim:
        emb = jnp.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb


def score(params, z, a_tau, tau, spec: ModelSpec, task=None):
    if z.ndim == 1:
        z = z[None]
    if a_tau.ndim == 2:
        a_tau = a_tau[None]
    z = task_emb(params, z, task, spec)
    a_flat = jnp.reshape(a_tau, (a_tau.shape[0], -1))
    t_emb = _time_embed(jnp.reshape(tau, (-1,)), spec.diffusion_time_embed_dim)
    inp = jnp.concatenate([z, a_flat, t_emb], axis=-1)
    out = apply_mlp(params["score"], inp, spec)
    return jnp.reshape(out, (a_tau.shape[0], spec.horizon, spec.action_dim))


def q_all_from_qs(qs_params, z, action, spec: ModelSpec, *, key=None, train=False):
    x = jnp.concatenate([z, action], axis=-1)
    keys = (
        jax.random.split(key, spec.num_q)
        if train and spec.dropout > 0.0 and key is not None
        else [None] * spec.num_q
    )
    return jnp.stack(
        [
            apply_mlp(qs_params[i], x, spec, dropout=spec.dropout, key=keys[i], train=train)
            for i in range(spec.num_q)
        ],
        axis=0,
    )


def q_all(params, z, action, spec: ModelSpec, *, key=None, train=False, task=None):
    z = task_emb(params, z, task, spec)
    return q_all_from_qs(params["qs"], z, action, spec, key=key, train=train)


def select_two_q_values(values, spec: ModelSpec, key=None):
    if spec.num_q > 2 and key is not None:
        idx = jax.random.permutation(key, spec.num_q)[:2]
        return jnp.take(values, idx, axis=0)
    return values[:2]


def q_value(params, z, action, spec: ModelSpec, return_type="avg", key=None, task=None):
    qs = q_all(params, z, action, spec, train=False, task=task)
    if return_type == "all":
        return qs
    values = jm.two_hot_inv(qs, spec.num_bins, spec.vmin, spec.vmax)
    if return_type == "min":
        return jnp.min(select_two_q_values(values, spec, key), axis=0)
    return jnp.mean(select_two_q_values(values, spec, key), axis=0)


def pi(params, z, key, spec: ModelSpec, deterministic=False, task=None):
    z = task_emb(params, z, task, spec)
    out = apply_mlp(params["pi"], z, spec)
    mean_raw, log_std_raw = jnp.split(out, 2, axis=-1)
    log_std = jm.log_std(log_std_raw, spec.log_std_min, spec.log_std_max)
    eps = (
        jnp.zeros_like(mean_raw)
        if deterministic
        else jax.random.normal(key, mean_raw.shape, dtype=mean_raw.dtype)
    )
    if spec.multitask:
        mask = action_mask(task, spec, mean_raw)
        mean_raw = mean_raw * mask
        log_std = log_std * mask
        eps = eps * mask
    pre_squash_log_prob = jm.gaussian_logprob(eps, log_std)
    scaled_log_prob = pre_squash_log_prob * action_size(task, spec, pre_squash_log_prob)
    action_raw = mean_raw + eps * jnp.exp(log_std)
    mean, action, log_prob = jm.squash(mean_raw, action_raw, pre_squash_log_prob)
    entropy_scale = scaled_log_prob / (log_prob + 1e-8)
    scaled_entropy = -log_prob * entropy_scale
    info = {
        "mean": mean,
        "log_std": log_std,
        "entropy": -log_prob,
        "scaled_entropy": scaled_entropy,
    }
    return action, info


def pi_mean(params, z, spec: ModelSpec, task=None):
    z = task_emb(params, z, task, spec)
    out = apply_mlp(params["pi"], z, spec)
    mean_raw, _ = jnp.split(out, 2, axis=-1)
    mean = jnp.tanh(mean_raw)
    if spec.multitask:
        mean = mean * action_mask(task, spec, mean)
    return mean
