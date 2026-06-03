import jax
import jax.numpy as jnp


LOG_2PI_HALF = 0.9189385175704956


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def two_hot(x, num_bins, vmin, vmax, bin_size):
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symlog(x)

    x = jnp.squeeze(jnp.clip(symlog(x), vmin, vmax), axis=-1)
    pos = (x - vmin) / bin_size
    lower = jnp.floor(pos).astype(jnp.int32)
    lower = jnp.clip(lower, 0, num_bins - 1)
    upper = (lower + 1) % num_bins
    offset = jnp.clip(pos - lower.astype(pos.dtype), 0.0, 1.0)
    return (
        jax.nn.one_hot(lower, num_bins) * (1.0 - offset)[..., None]
        + jax.nn.one_hot(upper, num_bins) * offset[..., None]
    )


def two_hot_inv(x, num_bins, vmin, vmax):
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symexp(x)
    bins = jnp.linspace(vmin, vmax, num_bins, dtype=x.dtype)
    probs = jax.nn.softmax(x, axis=-1)
    return symexp(jnp.sum(probs * bins, axis=-1, keepdims=True))


def soft_ce(pred, target, num_bins, vmin, vmax, bin_size):
    log_probs = jax.nn.log_softmax(pred, axis=-1)
    target_dist = two_hot(target, num_bins, vmin, vmax, bin_size)
    return -jnp.sum(target_dist * log_probs, axis=-1, keepdims=True)


def log_std(raw, low, high):
    return low + 0.5 * (high - low) * (jnp.tanh(raw) + 1.0)


def gaussian_logprob(eps, log_std_value):
    residual = -0.5 * jnp.square(eps) - log_std_value
    return jnp.sum(residual - LOG_2PI_HALF, axis=-1, keepdims=True)


def squash(mean, action, log_prob):
    mean = jnp.tanh(mean)
    action = jnp.tanh(action)
    correction = jnp.log(jnp.maximum(1.0 - jnp.square(action), 0.0) + 1e-6)
    log_prob = log_prob - jnp.sum(correction, axis=-1, keepdims=True)
    return mean, action, log_prob


def bce_with_logits(logits, labels):
    return jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(
        jnp.exp(-jnp.abs(logits))
    )
