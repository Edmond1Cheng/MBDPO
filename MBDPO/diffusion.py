import torch


class Diffusion:
    """No-gradient model-based diffusion planner for TD-MPC2."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._num_steps = max(int(cfg.diffusion_steps), 2)
        self._num_samples = int(cfg.diffusion_num_samples)
        self._temperature = float(cfg.diffusion_temperature)
        self._action_noise = float(cfg.diffusion_action_noise)
        self._num_elites = int(getattr(cfg, "diffusion_num_elites", 0) or 0)
        self._num_pi_trajs = int(getattr(cfg, "diffusion_num_pi_trajs", 0) or 0)

        self._eval_compile = bool(getattr(cfg, "diffusion_eval_compile", True))
        self._eval_compile_mode = str(
            getattr(cfg, "diffusion_eval_compile_mode", "reduce-overhead")
        )

        self._schedule_cache = {}
        self._compiled_eval_value_fn = None
        self._compiled_eval_agent_id = None

    def _get_value_fn(self, agent, eval_mode):
        """Return value estimation function, optionally compiled for eval-only hot path."""
        if not (eval_mode and self._eval_compile):
            return None
        agent_id = id(agent)
        if (
            self._compiled_eval_value_fn is None
            or self._compiled_eval_agent_id != agent_id
        ):

            def _value_fn(z, actions, task):
                return agent._estimate_value(z, actions, task)

            self._compiled_eval_value_fn = torch.compile(
                _value_fn, mode=self._eval_compile_mode
            )
            self._compiled_eval_agent_id = agent_id
        return self._compiled_eval_value_fn

    def _get_schedule(self, device):
        """Return cached (betas, alphas, alpha_bar) diffusion schedule for a device."""
        cache_key = str(device)
        if cache_key not in self._schedule_cache:
            betas = torch.linspace(
                self.cfg.diffusion_beta0,
                self.cfg.diffusion_betaT,
                self._num_steps,
                device=device,
            )
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            self._schedule_cache[cache_key] = (betas, alphas, alpha_bar)
        return self._schedule_cache[cache_key]

    @torch.no_grad()
    def plan(self, agent, obs, t0=False, eval_mode=False, task=None):
        cfg = self.cfg
        device = agent.device
        num_steps = self._num_steps
        num_samples = self._num_samples
        temperature = self._temperature
        action_noise = self._action_noise
        value_fn = self._get_value_fn(agent, eval_mode)

        z0 = agent.model.encode(obs, task)
        z = z0.repeat(num_samples, 1)

        _, alphas, alpha_bar = self._get_schedule(device)

        if t0:
            mean0 = torch.zeros(cfg.horizon, cfg.action_dim, device=device)
        else:
            mean0 = torch.cat(
                [
                    agent._prev_mean[1:],
                    torch.zeros_like(agent._prev_mean[:1]),
                ],
                dim=0,
            )

        x_tau = torch.sqrt(alpha_bar[-1]) * mean0
        action_mask = None
        if cfg.multitask:
            action_mask = agent.model._action_masks[task].squeeze(0)
            x_tau = x_tau * action_mask

        num_elites = self._num_elites
        num_pi_trajs = self._num_pi_trajs
        score_only_inference = bool(
            getattr(cfg, "score_only_inference", False)
        ) and bool(getattr(cfg, "use_score_network", False))

        if score_only_inference:
            for tau in range(num_steps - 1, 0, -1):
                alpha_bar_tau = alpha_bar[tau]
                score = agent.model.score(
                    z0, x_tau.unsqueeze(0), torch.tensor([tau], device=device), task
                ).squeeze(0)
                x_tau = (x_tau + (1.0 - alpha_bar_tau) * score) / torch.sqrt(
                    alphas[tau]
                )
                if action_mask is not None:
                    x_tau = x_tau * action_mask
        else:
            for tau in range(num_steps - 1, 0, -1):
                alpha_bar_tau = alpha_bar[tau]
                mean_cond = x_tau / torch.sqrt(alpha_bar_tau)
                std_cond = torch.sqrt((1.0 - alpha_bar_tau) / alpha_bar_tau)
                eps = torch.randn(
                    num_samples, cfg.horizon, cfg.action_dim, device=device
                )
                a0_samples = mean_cond.unsqueeze(0) + std_cond * eps
                if action_mask is not None:
                    a0_samples = a0_samples * action_mask
                a0_samples = a0_samples.clamp(-1, 1)

                if num_pi_trajs > 0:
                    pi_trajs = min(num_pi_trajs, num_samples)
                    pi_actions = torch.empty(
                        pi_trajs, cfg.horizon, cfg.action_dim, device=device
                    )
                    z_pi = z0.repeat(pi_trajs, 1)
                    for t in range(cfg.horizon):
                        a_pi, _ = agent.model.pi(z_pi, task)
                        if action_mask is not None:
                            a_pi = a_pi * action_mask
                        a_pi = a_pi.clamp(-1, 1)
                        pi_actions[:, t] = a_pi
                        z_pi = agent.model.next(z_pi, a_pi, task)
                    a0_samples[:pi_trajs] = pi_actions

                actions_for_value = a0_samples.permute(1, 0, 2)
                if value_fn is None:
                    values = agent._estimate_value(z, actions_for_value, task)
                else:
                    values = value_fn(z, actions_for_value, task)
                values = values.nan_to_num(0.0).squeeze(-1)

                g_mean = values.mean()
                g_std = values.std()
                if g_std < 1e-4:
                    g_std = torch.tensor(1.0, device=device)
                logits = (values - g_mean) / g_std
                logits = logits / max(temperature, 1e-6)
                if num_elites > 0 and num_elites < num_samples:
                    elite_idx = torch.topk(values, num_elites, dim=0).indices
                    elite_logits = logits[elite_idx]
                    elite_logits = elite_logits - elite_logits.max()
                    elite_weights = torch.softmax(elite_logits, dim=0)
                    a_bar = (elite_weights[:, None, None] * a0_samples[elite_idx]).sum(
                        dim=0
                    )
                else:
                    weights = torch.softmax(logits, dim=0)
                    a_bar = (weights[:, None, None] * a0_samples).sum(dim=0)

                score_mb = (-x_tau + torch.sqrt(alpha_bar_tau) * a_bar) / (
                    1.0 - alpha_bar_tau + 1e-8
                )
                score = (
                    agent.model.score(
                        z0, x_tau.unsqueeze(0), torch.tensor([tau], device=device), task
                    ).squeeze(0)
                    if bool(getattr(cfg, "use_score_network", False))
                    else score_mb
                )

                x_tau = (x_tau + (1.0 - alpha_bar_tau) * score) / torch.sqrt(
                    alphas[tau]
                )
                if action_mask is not None:
                    x_tau = x_tau * action_mask

        x0 = x_tau
        if action_mask is not None:
            x0 = x0 * action_mask
        x0 = x0.clamp(-1, 1)
        agent._prev_mean.copy_(x0.detach())

        action = x0[0]
        if not eval_mode:
            action = action + torch.randn_like(action) * action_noise
            if action_mask is not None:
                action = action * action_mask
        action = action.clamp(-1, 1)

        agent._last_imagined_z_traj = None
        return action
