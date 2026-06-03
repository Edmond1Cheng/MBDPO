import gymnasium as gym
import numpy as np

try:
    import torch
except ImportError:  # JAX-only environments do not need torch for env I/O.
    torch = None


class MultitaskWrapper(gym.Wrapper):
    """
    Wrapper for multi-task environments.
    """

    def __init__(self, cfg, envs):
        super().__init__(envs[0])
        self.cfg = cfg
        self.envs = envs
        self.backend = str(getattr(cfg, "backend", "torch"))
        if self.backend == "torch" and torch is None:
            raise ImportError("MultitaskWrapper backend='torch' requires torch.")
        self._task = cfg.tasks[0]
        self._task_idx = 0
        self._obs_shapes = [tuple(env.observation_space.shape) for env in self.envs]
        self._action_dims = [env.action_space.shape[0] for env in self.envs]
        self._episode_lengths = [env.max_episode_steps for env in self.envs]
        if len(self._obs_shapes[0]) == 1:
            self._obs_dims = [shape[0] for shape in self._obs_shapes]
            self._obs_shape = (max(self._obs_dims),)
            obs_low = -np.inf
            obs_high = np.inf
            obs_dtype = np.float32
        else:
            if any(shape != self._obs_shapes[0] for shape in self._obs_shapes):
                raise ValueError(
                    "Multi-task image observations require identical shapes across tasks; "
                    f"got {self._obs_shapes}."
                )
            self._obs_dims = [int(np.prod(shape)) for shape in self._obs_shapes]
            self._obs_shape = self._obs_shapes[0]
            obs_dtype = self.envs[0].observation_space.dtype
            obs_low = 0 if np.issubdtype(obs_dtype, np.integer) else -np.inf
            obs_high = 255 if np.issubdtype(obs_dtype, np.integer) else np.inf
        self._action_dim = max(self._action_dims)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=self._obs_shape, dtype=obs_dtype
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
        )

    @property
    def task(self):
        return self._task

    @property
    def task_idx(self):
        return self._task_idx

    @property
    def _env(self):
        return self.envs[self.task_idx]

    def rand_act(self):
        action = self.action_space.sample().astype(np.float32)
        return torch.from_numpy(action) if self.backend == "torch" else action

    def _pad_obs(self, obs):
        if tuple(obs.shape) == self._obs_shape:
            return obs
        if len(self._obs_shape) != 1:
            raise ValueError(
                f"Expected image observation shape {self._obs_shape}, got {tuple(obs.shape)}."
            )
        pad = self._obs_shape[0] - obs.shape[0]
        if pad < 0:
            raise ValueError(f"Observation shape {tuple(obs.shape)} exceeds {self._obs_shape}.")
        if self.backend == "torch":
            return torch.cat(
                (
                    obs,
                    torch.zeros(pad, dtype=obs.dtype, device=obs.device),
                )
            )
        return np.concatenate([obs, np.zeros(pad, dtype=obs.dtype)])

    def reset(self, task_idx=-1):
        self._task_idx = task_idx
        self._task = self.cfg.tasks[task_idx]
        self.env = self._env
        return self._pad_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(
            action[: self.env.action_space.shape[0]]
        )
        return self._pad_obs(obs), reward, done, info
