from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tensordict import TensorDict

from common import TASK_SET


def _infer_task_id(task_name: str, replay_task_id: int) -> int:
    if replay_task_id >= 0:
        return int(replay_task_id)
    for task_set in ("mt80", "mt30"):
        if task_name in TASK_SET[task_set]:
            return TASK_SET[task_set].index(task_name)
    return -1


def _expected_episode_length(task_name: str) -> int | None:
    if task_name.startswith("mw-"):
        return 101
    if task_name in TASK_SET["mt30"]:
        return 501
    return None


class ReplayBufferSaver:
    """Save online single-task episodes into TD-MPC2-style `.pt` chunks."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "save_replay", False))
        self._episodes = []
        self._chunk_idx = 0
        self._episodes_written = 0
        if not self.enabled:
            return
        root = Path(str(cfg.replay_save_dir))
        self._seed_dir = root / cfg.task / f"seed_{cfg.seed}"
        self._seed_dir.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(
            int(getattr(cfg, "replay_flush_every_episodes", 1000) or 1000), 1
        )
        self._include_terminated = bool(
            getattr(cfg, "replay_include_terminated", False)
        )
        self._task_id = _infer_task_id(
            cfg.task, int(getattr(cfg, "replay_task_id", -1) or -1)
        )
        self._episode_length = _expected_episode_length(cfg.task)

    def _prepare_episode(self, episode: TensorDict) -> TensorDict:
        episode = episode.to("cpu")
        if (
            self._episode_length is not None
            and int(episode.batch_size[0]) != self._episode_length
        ):
            raise ValueError(
                f'Expected task "{self.cfg.task}" episodes to have length {self._episode_length}, '
                f"got {int(episode.batch_size[0])}."
            )
        reward = episode["reward"].clone()
        action = episode["action"].clone()
        reward[0] = float("nan")
        action[0] = float("nan")
        payload = {
            "reward": reward,
            "obs": episode["obs"],
        }
        if self._task_id >= 0:
            payload["task"] = torch.full(
                reward.shape,
                self._task_id,
                dtype=torch.int32,
                device=torch.device("cpu"),
            )
        payload["action"] = action
        if self._include_terminated and "terminated" in episode.keys():
            terminated = episode["terminated"].clone()
            terminated[0] = float("nan")
            payload["terminated"] = terminated
        return TensorDict(
            payload, batch_size=episode.batch_size, device=torch.device("cpu")
        )

    def add_episode(self, episode: TensorDict):
        if not self.enabled:
            return
        prepared = self._prepare_episode(episode)
        if self._episodes:
            current_len = int(self._episodes[0].batch_size[0])
            incoming_len = int(prepared.batch_size[0])
            # Episodic environments (e.g., ManiSkill) can yield variable-length episodes.
            # TensorDict stack requires equal batch sizes, so flush before mixing lengths.
            if incoming_len != current_len:
                self.flush()
        self._episodes.append(prepared)
        if len(self._episodes) >= self._flush_every:
            self.flush()

    def flush(self):
        if not self.enabled or not self._episodes:
            return None
        batch = torch.stack(self._episodes, dim=0).contiguous()
        fp = self._seed_dir / f"chunk_{self._chunk_idx:05d}.pt"
        torch.save(batch, fp)
        self._episodes_written += batch.batch_size[0]
        self._chunk_idx += 1
        self._episodes.clear()
        return fp

    def finish(self):
        return self.flush()

    @property
    def episodes_written(self) -> int:
        return self._episodes_written


@dataclass
class MergeSummary:
    task: str
    num_files: int
    num_episodes: int
    episode_length: int
    output_path: Path


def merge_seed_replay_chunks(
    task: str, seed_dirs: Iterable[Path], output_path: Path, cleanup: bool = False
) -> MergeSummary:
    """Merge temporary per-seed replay chunks into a single task-level `.pt` file."""
    chunk_paths = []
    for seed_dir in seed_dirs:
        seed_dir = Path(seed_dir)
        chunk_paths.extend(sorted(seed_dir.glob("chunk_*.pt")))
    if not chunk_paths:
        raise FileNotFoundError(f'No replay chunks found for task "{task}".')

    parts_by_length: dict[int, list[TensorDict]] = {}
    num_episodes = 0
    episode_length = None
    for fp in chunk_paths:
        td = torch.load(fp, weights_only=False)
        if not isinstance(td, TensorDict):
            raise TypeError(f"Expected TensorDict at {fp}, got {type(td)}.")
        if len(td.batch_size) != 2:
            raise ValueError(
                f"Expected batch_size [num_episodes, episode_length] at {fp}, got {td.batch_size}."
            )
        length = int(td.batch_size[1])
        if episode_length is None:
            episode_length = length
        elif episode_length != length:
            episode_length = -1
        num_episodes += int(td.batch_size[0])
        parts_by_length.setdefault(length, []).append(td)

    merged_by_length = {}
    for length, length_parts in parts_by_length.items():
        merged_by_length[length] = (
            length_parts[0]
            if len(length_parts) == 1
            else torch.cat(length_parts, dim=0).contiguous()
        )
    if len(merged_by_length) == 1:
        merged = next(iter(merged_by_length.values()))
    else:
        # Keep variable-length episodes grouped by their sequence length.
        # This preserves full trajectories without padding artifacts.
        merged = {
            "variable_length": True,
            "task": task,
            "lengths": sorted(merged_by_length.keys()),
            "chunks_by_length": {
                str(k): v.to("cpu") for k, v in merged_by_length.items()
            },
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(merged, TensorDict):
        torch.save(merged.to("cpu"), output_path)
    else:
        torch.save(merged, output_path)

    if cleanup:
        for seed_dir in seed_dirs:
            seed_dir = Path(seed_dir)
            if seed_dir.exists():
                shutil.rmtree(seed_dir)

    return MergeSummary(
        task=task,
        num_files=len(chunk_paths),
        num_episodes=num_episodes,
        episode_length=int(episode_length),
        output_path=output_path,
    )
