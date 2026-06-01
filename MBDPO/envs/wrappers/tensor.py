from collections import defaultdict

import gymnasium as gym
import numpy as np


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for normalizing environment I/O to NumPy arrays.
    """

    def __init__(self, env):
        super().__init__(env)

    def rand_act(self):
        return self.action_space.sample().astype(np.float32)

    def _try_f32_array(self, x):
        if isinstance(x, np.ndarray):
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            return x
        return x

    def _obs_to_array(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_array(obs[k])
        else:
            obs = self._try_f32_array(obs)
        return obs

    def reset(self, task_idx=None):
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out
        return self._obs_to_array(obs)

    def step(self, action):
        step_out = self.env.step(np.asarray(action, dtype=np.float32))
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
            info = dict(info)
            info.setdefault("terminated", terminated)
        else:
            obs, reward, done, info = step_out
        info = defaultdict(float, info)
        info["success"] = float(info["success"])
        info["terminated"] = float(info["terminated"])
        return (
            self._obs_to_array(obs),
            np.float32(reward),
            done,
            info,
        )
