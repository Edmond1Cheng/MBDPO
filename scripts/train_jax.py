import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.getenv("PYOPENGL_PLATFORM", "egl")

import csv
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
import jax
import numpy as np
from termcolor import colored

from MBDPO.common import TASK_SET
from MBDPO.common.logger import Logger
from MBDPO.common.parser import parse_cfg
from MBDPO.common.seed import set_seed
from MBDPO.envs import make_env
from MBDPO.jax_impl import MBDPO
from MBDPO.jax_impl.common.buffer import Buffer
from MBDPO.jax_impl.parallel import (
    make_data_parallel_train_step,
    replicate_state,
    shard_batch,
)


SUCCESS_TASK_PREFIXES = ("mw-", "myo-")
MANISKILL2_TASKS = {"lift-cube", "pick-cube", "stack-cube", "pick-ycb", "turn-faucet"}


def _cfg_str(cfg, key, default=""):
    try:
        value = getattr(cfg, key)
    except Exception:
        return default
    if value is None:
        return default
    value = str(value).strip()
    if value in {"", "???", "None"}:
        return default
    return value


def _metrics_to_float(metrics):
    values = jax.device_get(metrics)
    return {k: float(np.asarray(v).mean()) for k, v in values.items()}


class RewardCsvWriter:
    """Small CSV writer for eval rewards/successes independent of wandb."""

    def __init__(self, cfg):
        self.enabled = bool(getattr(cfg, "save_reward_csv", False))
        self.freq = int(getattr(cfg, "csv_eval_freq", 50_000) or 50_000)
        self.mode = "eval_reward"
        self.fp = None
        if not self.enabled:
            return
        task = str(cfg.task)
        if task.startswith(SUCCESS_TASK_PREFIXES) or task in MANISKILL2_TASKS:
            self.mode = "eval_success"
        reward_csv_dir = Path(_cfg_str(cfg, "reward_csv_dir", "reward_csv"))
        reward_csv_dir.mkdir(parents=True, exist_ok=True)
        self.fp = reward_csv_dir / f"{task}_{cfg.seed}.csv"
        with open(self.fp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["step", "episode_success"]
                if self.mode == "eval_success"
                else ["step", "episode_reward"]
            )

    def _select_value(self, metrics):
        if self.mode == "eval_success":
            if "episode_success" in metrics:
                return metrics["episode_success"]
            values = [v for k, v in metrics.items() if k.startswith("episode_success+")]
            return float(np.nanmean(values)) if values else None
        if "episode_reward" in metrics:
            return metrics["episode_reward"]
        if "episode_reward+avg_dmcontrol" in metrics:
            return metrics["episode_reward+avg_dmcontrol"]
        values = [v for k, v in metrics.items() if k.startswith("episode_reward+")]
        return float(np.nanmean(values)) if values else None

    def append(self, step, metrics):
        if not self.enabled or self.fp is None:
            return
        if self.freq > 0 and int(step) % self.freq != 0:
            return
        value = self._select_value(metrics)
        if value is None or not np.isfinite(value):
            return
        with open(self.fp, "a", newline="") as f:
            csv.writer(f).writerow([int(step), float(value)])


class JaxUpdateEngine:
    """Single-device or pmap-backed update/act/save facade."""

    def __init__(self, cfg, agent):
        self.cfg = cfg
        self.agent = agent
        self.devices = []
        self.dp_state = None
        self.dp_step = None
        self._dirty = False
        requested = int(getattr(cfg, "jax_data_parallel_devices", 1) or 1)
        if requested <= 1:
            return
        available = jax.local_devices()
        if len(available) < requested:
            raise RuntimeError(
                f"Requested {requested} JAX devices but only found {len(available)}."
            )
        if int(cfg.batch_size) % requested != 0:
            raise ValueError(
                f"batch_size={cfg.batch_size} must be divisible by "
                f"jax_data_parallel_devices={requested}."
            )
        self.devices = available[:requested]
        self.dp_state = replicate_state(agent.state, self.devices)
        self.dp_step = make_data_parallel_train_step(
            agent.spec, agent.model_tx, agent.pi_tx, self.devices
        )
        print(
            colored("JAX data parallel devices:", "yellow", attrs=["bold"]),
            ", ".join(str(device) for device in self.devices),
            flush=True,
        )

    @property
    def data_parallel(self):
        return self.dp_step is not None

    def sync_to_single(self):
        if not self.data_parallel or not self._dirty:
            return
        self.agent.state = jax.tree_util.tree_map(
            lambda x: x[0], jax.device_get(self.dp_state)
        )
        self._dirty = False

    def update(self, batch):
        if not self.data_parallel:
            return self.agent.update(batch)
        sharded = shard_batch(batch, len(self.devices))
        self.dp_state, metrics = self.dp_step(self.dp_state, sharded)
        jax.block_until_ready(metrics)
        self._dirty = True
        return _metrics_to_float(metrics)

    def act(self, *args, **kwargs):
        self.sync_to_single()
        return self.agent.act(*args, **kwargs)

    def save(self, fp):
        self.sync_to_single()
        return self.agent.save(fp)

    def load(self, fp):
        self.agent.load(fp)
        if self.data_parallel:
            self.dp_state = replicate_state(self.agent.state, self.devices)
            self._dirty = False
        return self


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


def _reset_train_env(env, cfg, rng):
    if cfg.multitask:
        task_idx = int(rng.integers(0, len(cfg.tasks)))
        return _obs_array(env.reset(task_idx), cfg), task_idx
    return _obs_array(env.reset(), cfg), None


def _load_checkpoint_if_requested(agent, cfg):
    checkpoint = _cfg_str(cfg, "checkpoint")
    load_checkpoint = bool(getattr(cfg, "load_checkpoint", True))
    if checkpoint and load_checkpoint:
        print(colored("Loading checkpoint:", "yellow", attrs=["bold"]), checkpoint)
        agent.load(checkpoint)


def _common_step_metrics(step, episode, start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    return {
        "step": int(step),
        "episode": int(episode),
        "elapsed_time": elapsed,
        "steps_per_second": int(step) / elapsed if step else 0.0,
    }


class NpzOfflineSampler:
    """JAX/NumPy offline sampler.

    Expected arrays per .npz file:
      obs: [episodes, T+1, ...obs_shape] or [T+1, episodes, ...obs_shape]
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

        if obs.ndim < 3 or action.ndim < 3:
            raise ValueError(f"{fp} must contain episode/time batched obs and action arrays.")
        if obs.shape[1] in {action.shape[1], action.shape[1] + 1}:
            pass
        elif obs.shape[0] in {action.shape[0], action.shape[0] + 1}:
            obs = np.swapaxes(obs, 0, 1)
            action = np.swapaxes(action, 0, 1)
            reward = np.swapaxes(reward, 0, 1)
            terminated = np.swapaxes(terminated, 0, 1)
            if task is not None and task.ndim >= 2:
                task = np.swapaxes(task, 0, 1)
        else:
            raise ValueError(
                f"{fp} has incompatible obs/action shapes: {obs.shape} and {action.shape}."
            )

        if reward.ndim == 2:
            reward = reward[..., None]
        if terminated.ndim == 2:
            terminated = terminated[..., None]

        has_dummy_t0 = obs.shape[1] == action.shape[1]
        if not has_dummy_t0 and obs.shape[1] != action.shape[1] + 1:
            raise ValueError(
                f"{fp} must contain either standard obs[T+1], action[T] data "
                f"or TensorDict-style obs[T+1], action[T+1] data."
            )
        transition_count = action.shape[1] - 1 if has_dummy_t0 else action.shape[1]
        if transition_count < self.horizon:
            raise ValueError(f"{fp} is shorter than horizon={self.horizon}.")
        if "lengths" in data:
            lengths = np.asarray(data["lengths"], dtype=np.int32)
            lengths = np.minimum(lengths, transition_count)
        else:
            lengths = np.full((obs.shape[0],), transition_count, dtype=np.int32)
        if task is not None:
            task = self._episode_task_ids(task, obs.shape[0], fp)
        elif bool(getattr(self.cfg, "multitask", False)):
            raise ValueError(f"{fp} is missing required `task` ids for multitask offline training.")
        return {
            "obs": obs,
            "action": np.nan_to_num(action),
            "reward": np.nan_to_num(reward),
            "terminated": np.nan_to_num(terminated),
            "task": task,
            "lengths": lengths,
            "action_offset": 1 if has_dummy_t0 else 0,
        }

    def _episode_task_ids(self, task, num_episodes, fp):
        task = np.asarray(task, dtype=np.int32)
        if task.ndim == 0:
            return np.full((num_episodes,), int(task), dtype=np.int32)
        task = np.squeeze(task)
        if task.ndim == 1:
            if task.shape[0] != num_episodes:
                raise ValueError(
                    f"{fp} has task shape {task.shape}, expected one id per episode."
                )
            return task.astype(np.int32)
        if task.shape[0] != num_episodes:
            raise ValueError(
                f"{fp} has task shape {task.shape}, expected episode-major task ids."
            )
        return task[:, 0].astype(np.int32)

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
            length = int(part["lengths"][local_ep])
            max_start = length - self.horizon + 1
            start = int(self.rng.integers(0, max_start))
            end = start + self.horizon
            action_start = start + int(part["action_offset"])
            action_end = end + int(part["action_offset"])
            obs.append(part["obs"][local_ep, start : end + 1])
            action.append(part["action"][local_ep, action_start:action_end])
            reward.append(part["reward"][local_ep, action_start:action_end])
            terminated.append(part["terminated"][local_ep, action_start:action_end])
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
    if obs.ndim < 3 or action.ndim < 3:
        raise ValueError(f"{fp} must contain episode/time batched obs and action arrays.")
    if obs.shape[1] in {action.shape[1], action.shape[1] + 1}:
        obs_shape = tuple(int(x) for x in obs.shape[2:])
        transition_count = action.shape[1] - 1 if obs.shape[1] == action.shape[1] else action.shape[1]
    elif obs.shape[0] in {action.shape[0], action.shape[0] + 1}:
        obs_shape = tuple(int(x) for x in obs.shape[2:])
        transition_count = action.shape[0] - 1 if obs.shape[0] == action.shape[0] else action.shape[0]
    else:
        raise ValueError(
            f"{fp} has incompatible obs/action shapes: {obs.shape} and {action.shape}."
        )
    if cfg.obs == "state" and len(obs_shape) != 1:
        raise ValueError(f"{fp} contains obs shape {obs_shape}, but cfg.obs=state.")
    if cfg.obs == "rgb" and len(obs_shape) != 3:
        raise ValueError(f"{fp} contains obs shape {obs_shape}, but cfg.obs=rgb.")
    if "lengths" in data:
        episode_length = int(np.max(np.asarray(data["lengths"], dtype=np.int32)))
    else:
        episode_length = int(transition_count)
    action_dim = int(action.shape[-1])
    cfg.obs_shape = {str(cfg.obs): obs_shape}
    cfg.action_dim = action_dim
    cfg.episode_length = episode_length
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])
    cfg.action_dims = [action_dim for _ in cfg.tasks]
    cfg.episode_lengths = [episode_length for _ in cfg.tasks]


def train_online(cfg):
    env = make_env(cfg)
    raw_agent = MBDPO(cfg)
    _load_checkpoint_if_requested(raw_agent, cfg)
    agent = JaxUpdateEngine(cfg, raw_agent)
    buffer = Buffer(cfg)
    logger = Logger(cfg)
    reward_csv = RewardCsvWriter(cfg)
    seed_steps = int(getattr(cfg, "jax_seed_steps", cfg.seed_steps))
    updates_per_step = int(getattr(cfg, "jax_updates_per_step", 1) or 1)
    seed_pretrain = bool(getattr(cfg, "jax_seed_pretrain", True))
    log_freq = int(getattr(cfg, "jax_log_freq", 1000) or 1000)
    eval_freq = int(getattr(cfg, "eval_freq", 0) or 0)
    save_model_every = int(getattr(cfg, "save_model_every", 100_000) or 0)
    task_rng = np.random.default_rng(int(cfg.seed) + 17)

    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
    print(colored("JAX online task:", "yellow", attrs=["bold"]), cfg.task)
    print(colored("Seed steps:", "yellow", attrs=["bold"]), seed_steps)
    print(colored("Mode:", "yellow", attrs=["bold"]), "online")

    step, episode_idx = 0, 0
    start_time = time.time()
    obs, current_task = _reset_train_env(env, cfg, task_rng)
    if eval_freq > 0:
        eval_metrics = evaluate(agent, env, cfg, step=0)
        eval_metrics.update(_common_step_metrics(0, 0, start_time))
        reward_csv.append(0, eval_metrics)
        logger.log(eval_metrics, "eval")
        obs, current_task = _reset_train_env(env, cfg, task_rng)

    ep_obs = [obs]
    ep_actions, ep_rewards, ep_terminated = [], [], []
    train_metrics = {}
    next_eval_step = eval_freq if eval_freq > 0 else None

    while step < int(cfg.steps):
        if step > seed_steps:
            action = agent.act(
                obs,
                t0=(len(ep_actions) == 0),
                eval_mode=False,
                task=current_task,
            )
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
            _add_episode(
                buffer,
                ep_obs,
                ep_actions,
                ep_rewards,
                ep_terminated,
                task=current_task,
            )
            episode_idx += 1
            completed_step = step + 1
            train_log = dict(train_metrics)
            train_log.update(_common_step_metrics(completed_step, episode_idx, start_time))
            train_log.update(
                {
                    "episode_reward": float(np.sum(ep_rewards)),
                    "episode_success": float(info.get("success", 0.0)),
                    "episode_length": len(ep_rewards),
                    "episode_terminated": float(info.get("terminated", 0.0)),
                    "buffer_transitions": buffer.num_transitions,
                }
            )
            logger.log(train_log, "train")
            if next_eval_step is not None and completed_step >= next_eval_step:
                eval_metrics = evaluate(agent, env, cfg, step=completed_step)
                eval_metrics.update(
                    _common_step_metrics(completed_step, episode_idx, start_time)
                )
                reward_csv.append(completed_step, eval_metrics)
                logger.log(eval_metrics, "eval")
                while next_eval_step <= completed_step:
                    next_eval_step += eval_freq
            if (
                save_model_every > 0
                and completed_step > 0
                and completed_step % save_model_every == 0
            ):
                logger.save_agent(agent, identifier=f"{completed_step}")
            obs, current_task = _reset_train_env(env, cfg, task_rng)
            ep_obs = [obs]
            ep_actions, ep_rewards, ep_terminated = [], [], []

        if log_freq > 0 and step > 0 and step % log_freq == 0 and train_metrics:
            metrics = " ".join(
                f"{k}={v:.4f}" for k, v in sorted(train_metrics.items()) if np.isfinite(v)
            )
            print(f"update step={step} {metrics}", flush=True)

        step += 1

    _add_episode(
        buffer,
        ep_obs,
        ep_actions,
        ep_rewards,
        ep_terminated,
        task=current_task,
    )
    logger.finish(agent)
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
    raw_agent = MBDPO(cfg)
    _load_checkpoint_if_requested(raw_agent, cfg)
    agent = JaxUpdateEngine(cfg, raw_agent)
    logger = Logger(cfg)
    reward_csv = RewardCsvWriter(cfg)
    eval_freq = int(getattr(cfg, "eval_freq", 0) or 0)
    log_freq = int(getattr(cfg, "jax_log_freq", 10000) or 10000)
    save_model_every = int(getattr(cfg, "save_model_every", 100_000) or 0)
    start = time.time()
    print(colored("JAX offline task:", "yellow", attrs=["bold"]), cfg.task)
    print(colored("Offline episodes:", "yellow", attrs=["bold"]), sampler.num_eps)
    print(colored("Mode:", "yellow", attrs=["bold"]), "offline")
    for i in range(int(cfg.steps)):
        t0 = time.perf_counter()
        metrics = agent.update(sampler.sample())
        update_ms = 1000.0 * (time.perf_counter() - t0)
        log_due = i == 0 or (log_freq > 0 and (i + 1) % log_freq == 0)
        eval_due = eval_freq > 0 and i % eval_freq == 0
        save_due = save_model_every > 0 and i > 0 and i % save_model_every == 0
        if log_due or eval_due:
            log_metrics = dict(metrics)
            log_metrics.update(
                {
                    "iteration": int(i),
                    "elapsed_time": time.time() - start,
                    "update_ms": float(update_ms),
                }
            )
            if eval_due:
                eval_metrics = evaluate(agent, env, cfg, step=i)
                reward_csv.append(i, eval_metrics)
                log_metrics.update(eval_metrics)
                if cfg.multitask:
                    logger.pprint_multitask(log_metrics, cfg)
            logger.log(log_metrics, "pretrain")
        if save_due:
            logger.save_agent(agent, identifier=f"{i}")
    logger.finish(agent)
    print("JAX offline training completed successfully")


@hydra.main(config_name="config", config_path="../cfgs", version_base=None)
def main(cfg):
    cfg = parse_cfg(cfg)
    cfg.backend = "jax"
    set_seed(cfg.seed)
    if cfg.obs not in {"state", "rgb"}:
        raise ValueError("JAX implementation supports state and rgb observations only.")
    mode = str(getattr(cfg, "mode", "auto") or "auto").lower()
    if mode == "auto":
        mode = "offline" if cfg.multitask else "online"
    if mode not in {"online", "offline"}:
        raise ValueError("mode must be one of: auto, online, offline.")
    if mode == "offline":
        train_offline(cfg)
    else:
        train_online(cfg)


if __name__ == "__main__":
    main()
