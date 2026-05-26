from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from omegaconf import OmegaConf

from MBDPO.common import TASK_SET
from MBDPO.common.replay_buffer_saver import merge_seed_replay_chunks


def _resolve_tasks(cfg) -> list[str]:
    if cfg.get("tasks"):
        return list(cfg.tasks)
    task_set = str(cfg.get("task_set", "mt80"))
    if task_set not in TASK_SET:
        raise ValueError(
            f'Unknown task_set "{task_set}". Available: {list(TASK_SET.keys())}'
        )
    return list(TASK_SET[task_set])


def _resolve_gpu_ids(cfg) -> list[int]:
    if cfg.get("gpu_ids"):
        return [int(gpu_id) for gpu_id in cfg.gpu_ids]
    num_gpus = int(cfg.get("num_gpus", 0) or 0)
    if num_gpus <= 0:
        raise ValueError(
            "Please set either `gpu_ids` or `num_gpus > 0` in parallel_config.yaml."
        )
    return list(range(num_gpus))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _steps_for_task(cfg, task: str) -> int:
    if task.startswith("mw-"):
        return int(
            cfg.get("metaworld_steps", cfg.get("default_steps", cfg.get("steps", 0)))
            or 0
        )
    if task in TASK_SET["mt30"]:
        return int(
            cfg.get("dmc_steps", cfg.get("default_steps", cfg.get("steps", 0))) or 0
        )
    return int(cfg.get("default_steps", cfg.get("steps", 0)) or 0)


def _build_seed_command(
    cfg,
    task: str,
    seed: int,
    temp_replay_root: Path,
    reward_csv_root: Path,
    steps: int,
) -> list[str]:
    train_entry = str(cfg.get("train_entry", "scripts/train.py"))
    exp_name = str(cfg.get("exp_name", "parallel-online-diffusion"))
    cmd = [
        str(cfg.get("python_bin", sys.executable)),
        "-u",
        train_entry,
        f"task={task}",
        f"seed={seed}",
        f"steps={steps}",
        f'obs={cfg.get("obs", "state")}',
        f"exp_name={exp_name}",
        f'compile={str(bool(cfg.get("compile", False))).lower()}',
        f'compile_mode={cfg.get("compile_mode", "max-autotune-no-cudagraphs")}',
        f'diffusion_eval_compile={str(bool(cfg.get("diffusion_eval_compile", False))).lower()}',
        f'diffusion_eval_compile_mode={cfg.get("diffusion_eval_compile_mode", "reduce-overhead")}',
        f'eval_freq={int(cfg.get("eval_freq", 0) or 0)}',
        f'save_model_every={int(cfg.get("save_model_every", 0) or 0)}',
        f'save_agent={str(bool(cfg.get("save_agent", True))).lower()}',
        f'save_video={str(bool(cfg.get("save_video", False))).lower()}',
        f'enable_wandb={str(bool(cfg.get("enable_wandb", False))).lower()}',
        f'save_replay={str(bool(cfg.get("save_replay", True))).lower()}',
        f"replay_save_dir={temp_replay_root.as_posix()}",
        f'replay_flush_every_episodes={int(cfg.get("replay_flush_every_episodes", 1000))}',
        f'replay_include_terminated={str(bool(cfg.get("replay_include_terminated", True))).lower()}',
        f'save_reward_csv={str(bool(cfg.get("save_reward_csv", True))).lower()}',
        f"reward_csv_dir={reward_csv_root.as_posix()}",
        f'csv_eval_freq={int(cfg.get("csv_eval_freq", 50_000) or 50_000)}',
    ]
    for override in cfg.get("common_overrides", []) or []:
        cmd.append(str(override))
    wandb_name_template = str(cfg.get("wandb_name_template", "") or "").strip()
    if wandb_name_template:
        wandb_name = wandb_name_template.format(task=task, seed=seed)
        override_key = "wandb_name" if "wandb_name" in cfg else "+wandb_name"
        cmd.append(f"{override_key}={wandb_name}")
    return cmd


def _launch_seed_process(
    cfg,
    gpu_id: int,
    task: str,
    seed: int,
    temp_replay_root: Path,
    reward_csv_root: Path,
    log_root: Path,
):
    repo_root = _repo_root()
    steps = _steps_for_task(cfg, task)
    if steps <= 0:
        raise ValueError(f'Invalid step count {steps} for task "{task}".')
    seed_log_dir = log_root / task
    seed_log_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_seed_command(cfg, task, seed, temp_replay_root, reward_csv_root, steps)
    log_fp = seed_log_dir / f"seed_{seed}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(
        f'[gpu {gpu_id}] launching task={task} seed={seed} steps={steps}: {" ".join(cmd)}'
    )
    log_handle = open(log_fp, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return proc, log_fp, log_handle


def _wait_seed_process(
    seed: int, proc, log_fp: Path, log_handle, gpu_id: int, task: str
):
    return_code = proc.wait()
    log_handle.close()
    if return_code != 0:
        return seed, return_code, log_fp
    print(f"[gpu {gpu_id}] task={task} seed={seed} finished successfully. log={log_fp}")
    return None


def _merge_task_replay(cfg, task: str, temp_replay_root: Path, final_replay_root: Path):
    seed_dirs = [temp_replay_root / task / f"seed_{int(seed)}" for seed in cfg.seeds]
    output_path = final_replay_root / f"{task}.pt"
    summary = merge_seed_replay_chunks(
        task=task,
        seed_dirs=seed_dirs,
        output_path=output_path,
        cleanup=bool(cfg.get("merge_cleanup_temp", False)),
    )
    print(
        f"[merge] task={summary.task} files={summary.num_files} episodes={summary.num_episodes} "
        f"episode_length={summary.episode_length} output={summary.output_path}"
    )


def _resolve_job_mode(cfg) -> str:
    mode = str(cfg.get("job_mode", "task")).strip().lower()
    if mode not in {"task", "seed"}:
        raise ValueError(
            f'Invalid job_mode "{mode}". Expected one of ["task", "seed"].'
        )
    return mode


def _build_rotated_seed_job_queues(
    tasks: list[str], seeds: list[int], gpu_ids: list[int]
) -> dict[int, queue.Queue]:
    """
    Build per-GPU seed-job queues with seed-wise GPU rotation:
    - seed index 0 -> base task-to-gpu mapping by task index
    - seed index 1 -> shifted by +1 GPU
    - seed index 2 -> shifted by +2 GPU
    ...
    This spreads a task's seeds across different GPUs whenever possible.
    """
    if not gpu_ids:
        raise ValueError("No GPU IDs available for rotated seed scheduling.")
    per_gpu_queues = {gpu_id: queue.Queue() for gpu_id in gpu_ids}
    num_gpus = len(gpu_ids)
    for task_idx, task in enumerate(tasks):
        for seed_idx, seed in enumerate(seeds):
            target_gpu = gpu_ids[(task_idx + seed_idx) % num_gpus]
            per_gpu_queues[target_gpu].put((task, int(seed)))
    return per_gpu_queues


def _gpu_worker_task_mode(
    cfg,
    gpu_id: int,
    tasks_queue: queue.Queue,
    final_replay_root: Path,
    temp_replay_root: Path,
    reward_csv_root: Path,
    log_root: Path,
    failures: list,
):
    while True:
        try:
            task = tasks_queue.get_nowait()
        except queue.Empty:
            return
        try:
            start = time.time()
            shutil.rmtree(temp_replay_root / task, ignore_errors=True)
            failed = []
            parallel_seed_launch = bool(cfg.get("parallel_seeds_per_task", False))
            if parallel_seed_launch:
                procs = []
                for seed in cfg.seeds:
                    seed = int(seed)
                    procs.append(
                        (seed,)
                        + _launch_seed_process(
                            cfg,
                            gpu_id,
                            task,
                            seed,
                            temp_replay_root,
                            reward_csv_root,
                            log_root,
                        )
                    )
                for seed, proc, log_fp, log_handle in procs:
                    seed_failed = _wait_seed_process(
                        seed, proc, log_fp, log_handle, gpu_id, task
                    )
                    if seed_failed is not None:
                        failed.append(seed_failed)
            else:
                for seed in cfg.seeds:
                    seed = int(seed)
                    proc, log_fp, log_handle = _launch_seed_process(
                        cfg,
                        gpu_id,
                        task,
                        seed,
                        temp_replay_root,
                        reward_csv_root,
                        log_root,
                    )
                    seed_failed = _wait_seed_process(
                        seed, proc, log_fp, log_handle, gpu_id, task
                    )
                    if seed_failed is not None:
                        failed.append(seed_failed)
            if failed:
                failures.append((gpu_id, task, failed))
                print(f"[gpu {gpu_id}] task={task} failed: {failed}")
                return
            if bool(cfg.get("save_replay", True)):
                _merge_task_replay(cfg, task, temp_replay_root, final_replay_root)
            elapsed = time.time() - start
            print(f"[gpu {gpu_id}] task={task} completed in {elapsed/60:.1f} minutes.")
        finally:
            tasks_queue.task_done()


def _gpu_worker_seed_mode(
    cfg,
    gpu_id: int,
    seed_jobs_queue: queue.Queue,
    reward_csv_root: Path,
    temp_replay_root: Path,
    log_root: Path,
    failures: list,
):
    while True:
        try:
            task, seed = seed_jobs_queue.get_nowait()
        except queue.Empty:
            return
        try:
            start = time.time()
            proc, log_fp, log_handle = _launch_seed_process(
                cfg, gpu_id, task, seed, temp_replay_root, reward_csv_root, log_root
            )
            seed_failed = _wait_seed_process(
                seed, proc, log_fp, log_handle, gpu_id, task
            )
            if seed_failed is not None:
                failures.append((gpu_id, task, [seed_failed]))
                print(f"[gpu {gpu_id}] task={task} seed={seed} failed: {seed_failed}")
                return
            elapsed = time.time() - start
            print(
                f"[gpu {gpu_id}] task={task} seed={seed} completed in {elapsed/60:.1f} minutes."
            )
        finally:
            seed_jobs_queue.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="Launch parallel single-task online TD-MPC2 training jobs."
    )
    parser.add_argument(
        "--config",
        default=str(
            (
                Path(__file__).resolve().parents[1]
                / "cfgs"
                / "online_parallel_config.yaml"
            )
        ),
        help="Path to parallel training YAML config.",
    )
    parser.add_argument(
        "--replay_save_dir", default=None, help="Override replay_save_dir from config."
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None, help="Override num_gpus from config."
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None, help="Optional explicit task list override."
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.replay_save_dir is not None:
        cfg.replay_save_dir = args.replay_save_dir
    if args.num_gpus is not None:
        cfg.num_gpus = args.num_gpus
        cfg.gpu_ids = []
    if args.tasks:
        cfg.tasks = args.tasks
    tasks = _resolve_tasks(cfg)
    gpu_ids = _resolve_gpu_ids(cfg)
    job_mode = _resolve_job_mode(cfg)
    workers_per_gpu = int(cfg.get("workers_per_gpu", 1) or 1)
    if workers_per_gpu <= 0:
        raise ValueError("workers_per_gpu must be >= 1.")
    repo_root = _repo_root()
    final_replay_root = (
        repo_root / Path(str(cfg.get("replay_save_dir", "./replays")))
    ).resolve()
    temp_replay_root = final_replay_root / "_tmp"
    reward_csv_root = final_replay_root / "_csv"
    log_root = final_replay_root / "_logs"
    final_replay_root.mkdir(parents=True, exist_ok=True)
    temp_replay_root.mkdir(parents=True, exist_ok=True)
    reward_csv_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print(f"Using GPUs: {gpu_ids}")
    print(f"Tasks to run ({len(tasks)}): {tasks}")
    print(f"Per-task seeds: {[int(seed) for seed in cfg.seeds]}")
    print(f"Job mode: {job_mode}")
    print(f"Workers per GPU: {workers_per_gpu}")
    print(f"Final replay dir: {final_replay_root}")
    print(f"Temporary replay dir: {temp_replay_root}")
    print(f"Reward CSV dir: {reward_csv_root}")

    work_queue = queue.Queue()
    seed_job_queues = None
    if job_mode == "task":
        for task in tasks:
            work_queue.put(task)
    else:
        seed_job_queues = _build_rotated_seed_job_queues(
            tasks, [int(seed) for seed in cfg.seeds], gpu_ids
        )
        for gpu_id in gpu_ids:
            queued = list(seed_job_queues[gpu_id].queue)
            print(f"[seed-schedule] gpu={gpu_id} jobs={queued}")

    failures = []
    threads = []
    for gpu_id in gpu_ids:
        for _ in range(workers_per_gpu):
            if job_mode == "task":
                thread = threading.Thread(
                    target=_gpu_worker_task_mode,
                    args=(
                        cfg,
                        gpu_id,
                        work_queue,
                        final_replay_root,
                        temp_replay_root,
                        reward_csv_root,
                        log_root,
                        failures,
                    ),
                    daemon=False,
                )
            else:
                thread = threading.Thread(
                    target=_gpu_worker_seed_mode,
                    args=(
                        cfg,
                        gpu_id,
                        seed_job_queues[gpu_id],
                        reward_csv_root,
                        temp_replay_root,
                        log_root,
                        failures,
                    ),
                    daemon=False,
                )
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    if failures:
        print("Parallel training finished with failures:")
        for gpu_id, task, failed in failures:
            print(f"  gpu={gpu_id} task={task} failures={failed}")
        raise SystemExit(1)

    if bool(cfg.get("save_replay", True)):
        for task in tasks:
            _merge_task_replay(cfg, task, temp_replay_root, final_replay_root)

    print("Parallel training completed successfully.")


if __name__ == "__main__":
    main()
