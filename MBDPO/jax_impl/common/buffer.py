from collections import deque

import numpy as np


class Buffer:
    """Episode replay buffer for JAX training."""

    def __init__(self, cfg):
        self.horizon = int(cfg.horizon)
        self.batch_size = int(cfg.batch_size)
        self.capacity = int(min(cfg.buffer_size, cfg.steps))
        self._episodes = deque()
        self._transitions = 0
        self._rng = np.random.default_rng(int(cfg.seed))

    @property
    def num_episodes(self):
        return len(self._episodes)

    @property
    def num_transitions(self):
        return self._transitions

    def add_episode(self, obs, action, reward, terminated, task=None):
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1, 1)
        terminated = np.asarray(terminated, dtype=np.float32).reshape(-1, 1)
        if action.shape[0] < self.horizon:
            return
        if obs.shape[0] != action.shape[0] + 1:
            raise ValueError(
                f"Expected T+1 observations for T actions, got {obs.shape[0]} and {action.shape[0]}."
            )
        episode = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "task": None if task is None else int(task),
            "length": action.shape[0],
        }
        self._episodes.append(episode)
        self._transitions += episode["length"]
        while self._transitions > self.capacity and self._episodes:
            removed = self._episodes.popleft()
            self._transitions -= removed["length"]

    def can_sample(self):
        return any(ep["length"] >= self.horizon for ep in self._episodes)

    def sample(self):
        if not self.can_sample():
            raise RuntimeError("Replay buffer does not contain enough data to sample.")
        valid = [ep for ep in self._episodes if ep["length"] >= self.horizon]
        ep_indices = self._rng.integers(0, len(valid), size=self.batch_size)
        obs_batch, action_batch, reward_batch, terminated_batch, task_batch = [], [], [], [], []
        for ep_idx in ep_indices:
            ep = valid[int(ep_idx)]
            start = int(self._rng.integers(0, ep["length"] - self.horizon + 1))
            end = start + self.horizon
            obs_batch.append(ep["obs"][start : end + 1])
            action_batch.append(ep["action"][start:end])
            reward_batch.append(ep["reward"][start:end])
            terminated_batch.append(ep["terminated"][start:end])
            if ep["task"] is not None:
                task_batch.append(ep["task"])

        batch = {
            "obs": np.stack(obs_batch, axis=1),
            "action": np.stack(action_batch, axis=1),
            "reward": np.stack(reward_batch, axis=1),
            "terminated": np.stack(terminated_batch, axis=1),
        }
        if task_batch:
            batch["task"] = np.asarray(task_batch, dtype=np.int32)
        return batch
