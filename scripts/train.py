import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.getenv("PYOPENGL_PLATFORM", "egl")

import sys
import time
from glob import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "MBDPO"
for path in (str(PKG_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import hydra
import numpy as np
from termcolor import colored

from MBDPO import MBDPO
from MBDPO.common import TASK_SET
from MBDPO.common.buffer import Buffer
from MBDPO.common.parser import parse_cfg
from MBDPO.common.seed import set_seed
from MBDPO.envs import make_env


def _obs_array(obs, cfg):
    if isinstance(obs, dict):
        obs = obs[cfg.obs]
    return np.asarray(obs, dtype=np.float32)


def _env_action(action):
    return np.asarray(action, dtype=np.float32).copy()


def _add_episode(buffer, obs, actions, rewards, terminated, task=None):
    if not actions:
        return
    buffer.add_episode(
        np.stack(obs, axis=0),
        np.stack(actions, axis=0),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(terminated, dtype=np.float32),
        task=task,
    )


def _evaluate_single(agent, env, cfg, episodes, task_idx=None):
    rewards, successes, lengths = [], [], []
    for _ in range(int(episodes)):
        obs = env.reset(task_idx) if task_idx is not None else env.reset()
        obs = _obs_array(obs, cfg)
        done, ep_reward, t = False, 0.0, 0
        info = {}
        while not done:
            action = agent.act(
                obs,
                t0=(t == 0),
                eval_mode=True,
                task=task_idx,
            )
            obs, reward, done, info = env.step(_env_action(action))
            obs = _obs_array(obs, cfg)
            ep_reward += float(reward)
            t += 1
        rewards.append(ep_reward)
        successes.append(float(info.get("success", 0.0)))
        lengths.append(t)
    return (
        float(np.nanmean(rewards)),
        float(np.nanmean(successes)),
        float(np.nanmean(lengths)),
    )


def evaluate(agent, env, cfg, step):
    if env is None:
        return {}
    metrics = {}
    if cfg.multitask:
        for task_idx, task_name in enumerate(cfg.tasks):
            reward, success, length = _evaluate_single(
                agent, env, cfg, cfg.eval_episodes, task_idx=task_idx
            )
            metrics[f"episode_reward+{task_name}"] = reward
            metrics[f"episode_success+{task_name}"] = success
            metrics[f"episode_length+{task_name}"] = length
    else:
        reward, success, length = _evaluate_single(agent, env, cfg, cfg.eval_episodes)
        metrics = {
            "episode_reward": reward,
            "episode_success": success,
            "episode_length": length,
        }
    print(
        f"eval step={int(step)} "
        + " ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items())),
        flush=True,
    )
    return metrics


class NpzOfflineSampler:
    """JAX/NumPy offline sampler.

    Expected arrays per .npz file:
      obs: [episodes, T+1, obs_dim] or [T+1, episodes, obs_dim]
      action: [episodes, T, action_dim] or [T, episodes, action_dim]
      reward: matching [episodes, T] / [T, episodes], optional last dim 1
      terminated: optional, same shape as reward
      task: optional [episodes] integer task ids
    """

    def __init__(self, cfg, files):
        self.cfg = cfg
        self.horizon = int(cfg.horizon)
        self.batch_size = int(cfg.batch_size)
        self.rng = np.random.default_rng(int(cfg.seed))
        self.parts = [self._load_file(fp) for fp in files]
        self.episodes_per_part = [part["action"].shape[0] for part in self.parts]
        self.cum_eps = np.cumsum(self.episodes_per_part)
        self.num_eps = int(self.cum_eps[-1])

    def _load_file(self, fp):
        data = np.load(fp)
        obs = np.asarray(data["obs"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        reward = np.asarray(data["reward"], dtype=np.float32)
        terminated = np.asarray(
            data["terminated"] if "terminated" in data else np.zeros_like(reward),
            dtype=np.float32,
        )
        task = np.asarray(data["task"], dtype=np.int32) if "task" in data else None
        if obs.shape[0] == action.shape[0] + 1:
            obs = np.swapaxes(obs, 0, 1)
            action = np.swapaxes(action, 0, 1)
            reward = np.swapaxes(reward, 0, 1)
            terminated = np.swapaxes(terminated, 0, 1)
        if reward.ndim == 2:
            reward = reward[..., None]
        if terminated.ndim == 2:
            terminated = terminated[..., None]
        if obs.shape[1] != action.shape[1] + 1:
            raise ValueError(f"{fp} does not contain T+1 observations for T actions.")
        if action.shape[1] < self.horizon:
            raise ValueError(f"{fp} is shorter than horizon={self.horizon}.")
        return {
            "obs": obs,
            "action": np.nan_to_num(action),
            "reward": np.nan_to_num(reward),
            "terminated": np.nan_to_num(terminated),
            "task": task,
        }

    def _part_index(self, ep):
        part_idx = int(np.searchsorted(self.cum_eps, ep, side="right"))
        part_start = 0 if part_idx == 0 else int(self.cum_eps[part_idx - 1])
        return part_idx, int(ep - part_start)

    def sample(self):
        obs, action, reward, terminated, task = [], [], [], [], []
        eps = self.rng.integers(0, self.num_eps, size=self.batch_size)
        for ep in eps:
            part_idx, local_ep = self._part_index(int(ep))
            part = self.parts[part_idx]
            max_start = part["action"].shape[1] - self.horizon + 1
            start = int(self.rng.integers(0, max_start))
            end = start + self.horizon
            obs.append(part["obs"][local_ep, start : end + 1])
            action.append(part["action"][local_ep, start:end])
            reward.append(part["reward"][local_ep, start:end])
            terminated.append(part["terminated"][local_ep, start:end])
            if part["task"] is not None:
                task.append(int(part["task"][local_ep]))
        batch = {
            "obs": np.stack(obs, axis=1),
            "action": np.stack(action, axis=1),
            "reward": np.stack(reward, axis=1),
            "terminated": np.stack(terminated, axis=1),
        }
        if task:
            batch["task"] = np.asarray(task, dtype=np.int32)
        return batch


def _offline_files(data_dir):
    path = Path(str(data_dir))
    if path.is_file():
        files = [path]
    else:
        files = sorted(Path(p) for p in glob(str(path / "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz offline chunks found under {path}")
    return files


def _fill_cfg_from_npz(cfg, fp):
    data = np.load(fp)
    obs = np.asarray(data["obs"])
    action = np.asarray(data["action"])
    if obs.shape[0] == action.shape[0] + 1:
        obs_dim = int(obs.shape[-1])
        episode_length = int(action.shape[0])
    else:
        obs_dim = int(obs.shape[-1])
        episode_length = int(action.shape[1])
    action_dim = int(action.shape[-1])
    cfg.obs_shape = {"state": (obs_dim,)}
    cfg.action_dim = action_dim
    cfg.episode_length = episode_length
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    cfg.tasks = TASK_SET[cfg.task]
    cfg.action_dims = [action_dim for _ in cfg.tasks]
    cfg.episode_lengths = [episode_length for _ in cfg.tasks]


def train_online(cfg):
    env = make_env(cfg)
    agent = MBDPO(cfg)
    buffer = Buffer(cfg)
    seed_steps = int(getattr(cfg, "jax_seed_steps", cfg.seed_steps))
    updates_per_step = int(getattr(cfg, "jax_updates_per_step", 1) or 1)
    seed_pretrain = bool(getattr(cfg, "jax_seed_pretrain", True))
    log_freq = int(getattr(cfg, "jax_log_freq", 1000) or 1000)
    eval_freq = int(getattr(cfg, "eval_freq", 0) or 0)

    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
    print(colored("JAX online task:", "yellow", attrs=["bold"]), cfg.task)
    print(colored("Seed steps:", "yellow", attrs=["bold"]), seed_steps)

    step, episode_idx = 0, 0
    start_time = time.time()
    obs = _obs_array(env.reset(), cfg)
    if eval_freq > 0:
        evaluate(agent, env, cfg, step=0)
        obs = _obs_array(env.reset(), cfg)

    ep_obs = [obs]
    ep_actions, ep_rewards, ep_terminated = [], [], []
    train_metrics = {}
    next_eval_step = eval_freq if eval_freq > 0 else None

    while step < int(cfg.steps):
        if step > seed_steps:
            action = agent.act(obs, t0=(len(ep_actions) == 0), eval_mode=False)
        else:
            action = _obs_array(env.rand_act(), cfg)

        next_obs, reward, done, info = env.step(_env_action(action))
        next_obs = _obs_array(next_obs, cfg)
        ep_actions.append(action)
        ep_rewards.append(float(reward))
        ep_terminated.append(float(info.get("terminated", 0.0)))
        ep_obs.append(next_obs)
        obs = next_obs

        if step >= seed_steps and buffer.can_sample():
            num_updates = seed_steps if step == seed_steps and seed_pretrain else updates_per_step
            if num_updates == seed_steps:
                print("Pretraining JAX agent on seed data...", flush=True)
            for _ in range(num_updates):
                train_metrics = agent.update(buffer.sample())

        if done:
            _add_episode(buffer, ep_obs, ep_actions, ep_rewards, ep_terminated)
            episode_idx += 1
            completed_step = step + 1
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"step={completed_step} episode={episode_idx} "
                f"reward={sum(ep_rewards):.2f} length={len(ep_rewards)} "
                f"buffer={buffer.num_transitions} sps={completed_step / elapsed:.1f}",
                flush=True,
            )
            if next_eval_step is not None and completed_step >= next_eval_step:
                evaluate(agent, env, cfg, step=completed_step)
                while next_eval_step <= completed_step:
                    next_eval_step += eval_freq
            obs = _obs_array(env.reset(), cfg)
            ep_obs = [obs]
            ep_actions, ep_rewards, ep_terminated = [], [], []

        if log_freq > 0 and step > 0 and step % log_freq == 0 and train_metrics:
            metrics = " ".join(
                f"{k}={v:.4f}" for k, v in sorted(train_metrics.items()) if np.isfinite(v)
            )
            print(f"update step={step} {metrics}", flush=True)

        step += 1

    _add_episode(buffer, ep_obs, ep_actions, ep_rewards, ep_terminated)
    print("JAX online training completed successfully")


def train_offline(cfg):
    files = _offline_files(cfg.data_dir)
    env = None
    try:
        env = make_env(cfg)
    except Exception as exc:
        print(colored(f"Environment unavailable; using offline metadata only: {exc}", "yellow"))
        _fill_cfg_from_npz(cfg, files[0])
    sampler = NpzOfflineSampler(cfg, files)
    agent = MBDPO(cfg)
    eval_freq = int(getattr(cfg, "eval_freq", 0) or 0)
    log_freq = int(getattr(cfg, "jax_log_freq", 10000) or 10000)
    start = time.time()
    print(colored("JAX offline task:", "yellow", attrs=["bold"]), cfg.task)
    print(colored("Offline episodes:", "yellow", attrs=["bold"]), sampler.num_eps)
    for i in range(int(cfg.steps)):
        t0 = time.perf_counter()
        metrics = agent.update(sampler.sample())
        update_ms = 1000.0 * (time.perf_counter() - t0)
        log_due = i == 0 or (log_freq > 0 and (i + 1) % log_freq == 0)
        eval_due = eval_freq > 0 and i % eval_freq == 0
        if log_due or eval_due:
            text = " ".join(
                f"{k}={v:.6f}" for k, v in sorted(metrics.items()) if np.isfinite(v)
            )
            print(
                f"offline iter={i} elapsed={time.time() - start:.1f}s "
                f"update_ms={update_ms:.3f} {text}",
                flush=True,
            )
            if eval_due:
                evaluate(agent, env, cfg, step=i)
    print("JAX offline training completed successfully")


@hydra.main(config_name="config", config_path="../cfgs", version_base=None)
def main(cfg):
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    if cfg.obs not in {"state", "rgb"}:
        raise ValueError("JAX implementation supports state and rgb observations only.")
    if cfg.multitask:
        train_offline(cfg)
    else:
        train_online(cfg)


if __name__ == "__main__":
    main()
