import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from common.mmap_buffer import MmapBuffer
from trainer.base import Trainer


class OfflineTrainer(Trainer):
    """Trainer class for multi-task offline TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        results = dict()
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating"):
            ep_rewards, ep_successes = [], []
            for _ in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
                while not done:
                    action = self.agent.act(
                        obs, t0=t == 0, eval_mode=True, task=task_idx
                    )
                    obs, reward, done, info = self.env.step(action)
                    ep_reward += reward
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(info["success"])
            results.update(
                {
                    f"episode_reward+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_rewards
                    ),
                    f"episode_success+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_successes
                    ),
                }
            )
        return results

    def _load_dataset(self):
        """Load dataset for offline training."""
        fp = Path(os.path.join(self.cfg.data_dir, "*.pt"))
        fps = sorted(glob(str(fp)))
        assert len(fps) > 0, f"No data found at {fp}"
        print(f"Found {len(fps)} files in {fp}")
        if len(fps) < (20 if self.cfg.task == "mt80" else 4):
            print(
                f"WARNING: expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files."
            )

        # Create buffer for sampling
        _cfg = deepcopy(self.cfg)
        _cfg.episode_length = 101 if self.cfg.task == "mt80" else 501
        _cfg.buffer_size = 550_450_000 if self.cfg.task == "mt80" else 345_690_000
        _cfg.steps = _cfg.buffer_size
        data_mode = str(getattr(self.cfg, "offline_data_mode", "in_memory"))
        if data_mode == "mmap":
            print(
                "Using mmap/lazy offline dataset mode (shared file-backed pages across processes when supported)."
            )
            parts = []
            for fp in tqdm(fps, desc="Loading data (mmap)"):
                try:
                    td = torch.load(
                        fp, weights_only=False, map_location="cpu", mmap=True
                    )
                except TypeError:
                    td = torch.load(fp, weights_only=False, map_location="cpu")
                assert td.shape[1] == _cfg.episode_length, (
                    f"Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, "
                    f"please double-check your config."
                )
                parts.append(td)
            self.buffer = MmapBuffer(_cfg, parts, episode_length=_cfg.episode_length)
        else:
            self.buffer = Buffer(_cfg)
            for fp in tqdm(fps, desc="Loading data"):
                td = torch.load(fp, weights_only=False)
                assert td.shape[1] == _cfg.episode_length, (
                    f"Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, "
                    f"please double-check your config."
                )
                self.buffer.load(td)
        expected_episodes = _cfg.buffer_size // _cfg.episode_length
        if self.buffer.num_eps != expected_episodes:
            print(
                f"WARNING: buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes for {self.cfg.task} task set."
            )

    def train(self):
        """Train a TD-MPC2 agent."""
        assert self.cfg.multitask and self.cfg.task in {
            "mt30",
            "mt80",
        }, "Offline training only supports multitask training with mt30 or mt80 task sets."
        self._load_dataset()

        print(f"Training agent for {self.cfg.steps} iterations...")
        metrics = {}
        eval_freq = int(getattr(self.cfg, "eval_freq", 0) or 0)
        save_model_every = int(getattr(self.cfg, "save_model_every", 100_000) or 0)
        for i in range(self.cfg.steps):

            # Update agent
            train_metrics = self.agent.update(self.buffer)
            eval_due = eval_freq > 0 and i % eval_freq == 0
            save_due = save_model_every > 0 and i > 0 and i % save_model_every == 0
            log_due = eval_due or i % 10_000 == 0

            # Evaluate agent periodically
            if log_due:
                metrics = {
                    "iteration": i,
                    "elapsed_time": time() - self._start_time,
                }
                metrics.update(train_metrics)
                if eval_due:
                    metrics.update(self.eval())
                    self.logger.pprint_multitask(metrics, self.cfg)
                self.logger.log(metrics, "pretrain")

            # Save model periodically (independent from eval)
            if save_due:
                self.logger.save_agent(self.agent, identifier=f"{i}")

        self.logger.finish(self.agent)
