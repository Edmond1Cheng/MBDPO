from collections import defaultdict

import gymnasium as gym
import numpy as np

try:
    import torch
except ImportError:  # JAX-only environments do not need torch for env I/O.
    torch = None


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for normalizing environment I/O.

    The default backend preserves the Torch training path. JAX callers set
    `cfg.backend = "jax"` and receive NumPy arrays/scalars from the same envs.
    """

    def __init__(self, env, backend="torch"):
        super().__init__(env)
        self.backend = str(backend)
        if self.backend == "torch" and torch is None:
            raise ImportError("TensorWrapper backend='torch' requires torch.")

    def rand_act(self):
        action = self.action_space.sample().astype(np.float32)
        return torch.from_numpy(action) if self.backend == "torch" else action

    def _try_f32_array(self, x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32) if x.dtype == np.float64 else x
        return x

    def _try_f32_obs(self, x):
        if self.backend == "jax":
            return self._try_f32_array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dtype == torch.float64:
                x = x.float()
        return x

    def _obs_to_backend(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_obs(obs[k])
        else:
            obs = self._try_f32_obs(obs)
        return obs

    def reset(self, task_idx=None):
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out
        return self._obs_to_backend(obs)

    def _action_to_numpy(self, action):
        if torch is not None and torch.is_tensor(action):
            return action.detach().cpu().numpy()
        return np.asarray(action, dtype=np.float32)

    def step(self, action):
        step_out = self.env.step(self._action_to_numpy(action))
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
            info = dict(info)
            info.setdefault("terminated", terminated)
        else:
            obs, reward, done, info = step_out
        info = defaultdict(float, info)
        info["success"] = float(info["success"])
        terminated = float(info["terminated"])
        if self.backend == "jax":
            info["terminated"] = terminated
            return self._obs_to_backend(obs), np.float32(reward), done, info
        info["terminated"] = torch.tensor(terminated)
        return self._obs_to_backend(obs), torch.tensor(reward, dtype=torch.float32), done, info
