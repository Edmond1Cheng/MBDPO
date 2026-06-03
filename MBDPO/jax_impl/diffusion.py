import jax
import jax.numpy as jnp

from .common import world_model as wm


def make_plan_step(spec: wm.ModelSpec, estimate_value_fn):
    """Build the JIT-compiled diffusion planner used by the JAX agent."""
    betas = jnp.linspace(spec.diffusion_beta0, spec.diffusion_betaT, spec.diffusion_steps)
    alphas = 1.0 - betas
    alpha_bar = jnp.cumprod(alphas)

    @jax.jit
    def plan_step(
        params,
        prev_mean,
        key,
        obs,
        t0,
        eval_mode,
        contrastive_mean,
        contrastive_std,
        task,
    ):
        key, k_encode, k_loop, k_action = jax.random.split(key, 4)
        task_arg = task if spec.multitask else None
        z0 = wm.encode(params, obs[None], spec, task=task_arg, key=k_encode)
        z = jnp.repeat(z0, spec.diffusion_num_samples, axis=0)
        shifted = jnp.concatenate(
            [prev_mean[1:], jnp.zeros_like(prev_mean[:1])], axis=0
        )
        mean0 = jnp.where(t0, jnp.zeros_like(prev_mean), shifted)
        x_tau = jnp.sqrt(alpha_bar[-1]) * mean0
        if spec.multitask:
            x_tau = x_tau * wm.action_mask(task_arg, spec, x_tau)

        def loop_body(i, carry):
            x_t, loop_key = carry
            tau = spec.diffusion_steps - 1 - i
            loop_key, k_eps, k_value, k_pi = jax.random.split(loop_key, 4)
            alpha_bar_tau = alpha_bar[tau]
            if spec.score_only_inference and spec.use_score_network:
                score = wm.score(
                    params,
                    z0,
                    x_t[None],
                    jnp.asarray([tau], dtype=jnp.int32),
                    spec,
                    task=task_arg,
                )[0]
            else:
                mean_cond = x_t / jnp.sqrt(alpha_bar_tau)
                std_cond = jnp.sqrt((1.0 - alpha_bar_tau) / alpha_bar_tau)
                eps = jax.random.normal(
                    k_eps,
                    (spec.diffusion_num_samples, spec.horizon, spec.action_dim),
                    dtype=x_t.dtype,
                )
                a0_samples = jnp.clip(mean_cond[None] + std_cond * eps, -1.0, 1.0)
                if spec.multitask:
                    a0_samples = a0_samples * wm.action_mask(task_arg, spec, a0_samples)

                if spec.diffusion_num_pi_trajs > 0:
                    pi_trajs = min(
                        spec.diffusion_num_pi_trajs, spec.diffusion_num_samples
                    )
                    pi_keys = jax.random.split(k_pi, spec.horizon)

                    def pi_body(z_pi, pi_key):
                        a_pi, _ = wm.pi(params, z_pi, pi_key, spec, task=task_arg)
                        a_pi = jnp.clip(a_pi, -1.0, 1.0)
                        z_next = wm.next_latent(params, z_pi, a_pi, spec, task=task_arg)
                        return z_next, a_pi

                    z_pi0 = jnp.repeat(z0, pi_trajs, axis=0)
                    _, pi_actions = jax.lax.scan(pi_body, z_pi0, pi_keys)
                    pi_actions = jnp.swapaxes(pi_actions, 0, 1)
                    a0_samples = a0_samples.at[:pi_trajs].set(pi_actions)

                values = estimate_value_fn(
                    params,
                    z,
                    jnp.swapaxes(a0_samples, 0, 1),
                    spec,
                    contrastive_mean,
                    contrastive_std,
                    k_value,
                    task=task_arg,
                )
                values = jnp.nan_to_num(values[:, 0], nan=0.0)
                raw_value_std = jnp.std(values, ddof=1)
                value_std = jnp.where(raw_value_std < 1e-4, 1.0, raw_value_std)
                logits = (values - jnp.mean(values)) / value_std
                logits = logits / max(spec.diffusion_temperature, 1e-6)
                if (
                    spec.diffusion_num_elites > 0
                    and spec.diffusion_num_elites < spec.diffusion_num_samples
                ):
                    _, elite_idx = jax.lax.top_k(values, spec.diffusion_num_elites)
                    elite_logits = logits[elite_idx] - jnp.max(logits[elite_idx])
                    weights = jax.nn.softmax(elite_logits, axis=0)
                    a_bar = jnp.sum(
                        weights[:, None, None] * a0_samples[elite_idx], axis=0
                    )
                else:
                    weights = jax.nn.softmax(logits, axis=0)
                    a_bar = jnp.sum(weights[:, None, None] * a0_samples, axis=0)

                score_mb = (-x_t + jnp.sqrt(alpha_bar_tau) * a_bar) / (
                    1.0 - alpha_bar_tau + 1e-8
                )
                score = (
                    wm.score(
                        params,
                        z0,
                        x_t[None],
                        jnp.asarray([tau], dtype=jnp.int32),
                        spec,
                        task=task_arg,
                    )[0]
                    if spec.use_score_network
                    else score_mb
                )
            x_next = (x_t + (1.0 - alpha_bar_tau) * score) / jnp.sqrt(alphas[tau])
            if spec.multitask:
                x_next = x_next * wm.action_mask(task_arg, spec, x_next)
            return (x_next, loop_key)

        x0, _ = jax.lax.fori_loop(
            0, spec.diffusion_steps - 1, loop_body, (x_tau, k_loop)
        )
        x0 = jnp.clip(x0, -1.0, 1.0)
        if spec.multitask:
            x0 = x0 * wm.action_mask(task_arg, spec, x0)
        action = x0[0]
        noise = jax.random.normal(k_action, action.shape, dtype=action.dtype)
        action = jnp.where(
            eval_mode,
            action,
            jnp.clip(action + spec.diffusion_action_noise * noise, -1.0, 1.0),
        )
        if spec.multitask:
            action = action * wm.action_mask(task_arg, spec, action)
        return action, x0, key

    return plan_step
