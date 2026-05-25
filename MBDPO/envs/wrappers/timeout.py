import gymnasium as gym


class Timeout(gym.Wrapper):
    """
    Wrapper for enforcing a time limit on the environment.
    """

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.max_episode_steps_value = max_episode_steps

    @property
    def max_episode_steps(self):
        return self.max_episode_steps_value

    def reset(self, **kwargs):
        self.elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
            info = dict(info)
            info.setdefault("terminated", bool(terminated))
        else:
            obs, reward, done, info = step_out
            info = dict(info)
            info.setdefault("terminated", bool(done))
        self.elapsed_steps += 1
        timeout = self.elapsed_steps >= self.max_episode_steps
        done = done or timeout
        if timeout:
            # Time-limit endings are truncations, not environment terminations.
            info["terminated"] = False
            info["truncated"] = True
        return obs, reward, done, info
