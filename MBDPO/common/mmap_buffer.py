import torch


class MmapBuffer:
    """
    Lazy/mmap-backed offline buffer for multi-task training.

    Instead of materializing the entire replay into a ReplayBuffer, this class keeps
    file-backed TensorDicts (when supported by `torch.load(..., mmap=True)`) and
    samples short horizon slices on demand.
    """

    def __init__(self, cfg, dataset_parts, episode_length):
        self.cfg = cfg
        self._device = torch.device("cuda:0")
        self._dataset_parts = dataset_parts
        self._episode_length = int(episode_length)
        self._horizon = int(cfg.horizon)
        self._batch_size = int(cfg.batch_size)
        self._slice_len = self._horizon + 1
        if self._slice_len > self._episode_length:
            raise ValueError(
                f"horizon+1 ({self._slice_len}) cannot exceed episode_length ({self._episode_length})."
            )
        self._max_start = self._episode_length - self._slice_len + 1
        self._episodes_per_part = [int(td.shape[0]) for td in self._dataset_parts]
        self._num_eps = int(sum(self._episodes_per_part))
        if self._num_eps <= 0:
            raise ValueError("MmapBuffer requires at least one episode.")
        self._cum_eps = []
        cum = 0
        for n in self._episodes_per_part:
            cum += n
            self._cum_eps.append(cum)
        self._cum_eps_t = torch.tensor(self._cum_eps, dtype=torch.long)
        print(
            f"[MmapBuffer] parts={len(self._dataset_parts)} episodes={self._num_eps} "
            f"episode_length={self._episode_length} slice_len={self._slice_len}"
        )

    @property
    def num_eps(self):
        return self._num_eps

    def _global_to_part(self, global_ep_idx):
        part_idx = int(
            torch.searchsorted(self._cum_eps_t, global_ep_idx, right=True).item()
        )
        part_start = 0 if part_idx == 0 else self._cum_eps[part_idx - 1]
        local_ep_idx = int(global_ep_idx.item()) - part_start
        return part_idx, local_ep_idx

    def _prepare_batch(self, td):
        td = td.select(
            "obs", "action", "reward", "terminated", "task", strict=False
        ).to(self._device, non_blocking=True)
        obs = td.get("obs").contiguous()
        action = td.get("action")[1:].contiguous()
        reward = td.get("reward")[1:].unsqueeze(-1).contiguous()
        terminated = td.get("terminated", None)
        if terminated is not None:
            terminated = td.get("terminated")[1:].unsqueeze(-1).contiguous()
        else:
            terminated = torch.zeros_like(reward)
        task = td.get("task", None)
        if task is not None:
            task = task[0].contiguous()
        return obs, action, reward, terminated, task

    def sample(self):
        starts = torch.randint(
            0, self._max_start, size=(self._batch_size,), dtype=torch.long
        )
        global_eps = torch.randint(
            0, self._num_eps, size=(self._batch_size,), dtype=torch.long
        )
        slices = []
        for i in range(self._batch_size):
            part_idx, local_ep_idx = self._global_to_part(global_eps[i])
            start = int(starts[i].item())
            stop = start + self._slice_len
            episode_td = self._dataset_parts[part_idx][local_ep_idx]
            slices.append(episode_td[start:stop])
        td = torch.stack(slices, dim=0).permute(1, 0)
        return self._prepare_batch(td)
