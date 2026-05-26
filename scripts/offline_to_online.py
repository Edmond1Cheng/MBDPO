import csv
import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
from copy import deepcopy

import hydra
import torch
from termcolor import colored

from MBDPO.common import TASK_SET
from MBDPO.common.buffer import Buffer
from MBDPO.common.parser import parse_cfg
from MBDPO.common.seed import set_seed
from MBDPO.envs import make_env
from MBDPO import MBDPO

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


SUPPORTED_TASKS = {
    "walker-run": "walker-run",
    "walker run": "walker-run",
    "cheetah-run": "cheetah-run",
    "cheetah run": "cheetah-run",
    "hopper-hop": "hopper-hop",
    "hopper hop": "hopper-hop",
    "pendulum-swingup": "pendulum-swingup",
    "pendulum swingup": "pendulum-swingup",
    "reacher-hard": "reacher-hard",
    "reacher hard": "reacher-hard",
    "mw-bin-picking": "mw-bin-picking",
    "bin picking": "mw-bin-picking",
    "mw-box-close": "mw-box-close",
    "box close": "mw-box-close",
    "mw-door-lock": "mw-door-lock",
    "door lock": "mw-door-lock",
    "mw-door-unlock": "mw-door-unlock",
    "door unlock": "mw-door-unlock",
    "mw-hand-insert": "mw-hand-insert",
    "hand insert": "mw-hand-insert",
}

SOURCE_TASK_MAP = {
    "walker-run": "walker-walk",
    "cheetah-run": "cheetah-run-backwards",
    "hopper-hop": "hopper-stand",
    "pendulum-swingup": "pendulum-spin",
    "reacher-hard": "reacher-easy",
    "mw-bin-picking": "mw-pick-place",
    "mw-box-close": "mw-assembly",
    "mw-door-lock": "mw-door-open",
    "mw-door-unlock": "mw-door-open",
    "mw-hand-insert": "mw-sweep-into",
}


def _canonical_task_name(raw_name: str) -> str:
    key = str(raw_name).strip().lower().replace("_", "-").replace("  ", " ")
    if key in SUPPORTED_TASKS:
        return SUPPORTED_TASKS[key]
    raise ValueError(
        f'Unsupported offline-to-online task "{raw_name}". '
        f"Supported: {sorted(set(SUPPORTED_TASKS.values()))}"
    )


def _load_checkpoint_state_dict(checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return state["model"] if isinstance(state, dict) and "model" in state else state


def _infer_mt80_dims(state_dict: dict, task_dim: int):
    if "_action_masks" not in state_dict:
        raise KeyError(
            "Checkpoint must contain `_action_masks` for mt80 multitask model."
        )
    action_dim = int(state_dict["_action_masks"].shape[1])
    if "_encoder.state.0.weight" not in state_dict:
        raise KeyError(
            "Checkpoint must contain `_encoder.state.0.weight` to infer state observation dimension."
        )
    obs_plus_task = int(state_dict["_encoder.state.0.weight"].shape[1])
    obs_dim = obs_plus_task - int(task_dim)
    if obs_dim <= 0:
        raise ValueError(
            f"Invalid inferred obs_dim={obs_dim}; check checkpoint and task_dim={task_dim}."
        )
    action_dims = state_dict["_action_masks"].sum(-1).to(torch.int64).tolist()
    return obs_dim, action_dim, action_dims


def _default_mt80_dims(cfg):
    obs_dim = int(getattr(cfg, "mt80_obs_dim", 39) or 39)
    action_dim = int(getattr(cfg, "mt80_action_dim", 7) or 7)
    action_dims_cfg = getattr(cfg, "mt80_action_dims", None)
    action_dims = []
    if action_dims_cfg is not None:
        try:
            action_dims = [int(x) for x in action_dims_cfg]
        except TypeError:
            action_dims = []
    if len(action_dims) != len(TASK_SET["mt80"]):
        action_dims = [action_dim for _ in TASK_SET["mt80"]]
    return obs_dim, action_dim, action_dims


def _episode_length(task_name: str) -> int:
    return 100 if task_name.startswith("mw-") else 500


def _rebuild_optimizers(agent):
    agent.optim = torch.optim.Adam(
        [
            {
                "params": agent.model._encoder.parameters(),
                "lr": agent.cfg.lr * agent.cfg.enc_lr_scale,
            },
            {"params": agent.model._dynamics.parameters()},
            {"params": agent.model._reward.parameters()},
            {
                "params": (
                    agent.model._termination.parameters() if agent.cfg.episodic else []
                )
            },
            {"params": agent.model._F.parameters()},
            {"params": agent.model._Qs.parameters()},
            {
                "params": (
                    agent.model._task_emb.parameters() if agent.cfg.multitask else []
                )
            },
        ],
        lr=agent.cfg.lr,
        capturable=True,
    )
    agent.pi_optim = torch.optim.Adam(
        agent.model._pi.parameters(), lr=agent.cfg.lr, eps=1e-5, capturable=True
    )


def _append_new_task_embedding(agent, target_task: str, source_task: str):
    source_idx = agent.cfg.tasks.index(source_task)
    old_emb = agent.model._task_emb
    new_emb = torch.nn.Embedding(
        old_emb.num_embeddings + 1, old_emb.embedding_dim, max_norm=1
    ).to(old_emb.weight.device)
    with torch.no_grad():
        new_emb.weight[:-1].copy_(old_emb.weight)
        new_emb.weight[-1].copy_(old_emb.weight[source_idx])
    agent.model._task_emb = new_emb

    old_masks = agent.model._action_masks
    new_masks = torch.zeros(
        old_masks.shape[0] + 1,
        old_masks.shape[1],
        device=old_masks.device,
        dtype=old_masks.dtype,
    )
    new_masks[:-1].copy_(old_masks)
    new_masks[-1].copy_(old_masks[source_idx])
    delattr(agent.model, "_action_masks")
    agent.model.register_buffer("_action_masks", new_masks)

    agent.cfg.tasks = list(agent.cfg.tasks) + [f"off2on-{target_task}"]
    agent.cfg.action_dims = list(agent.cfg.action_dims) + [
        int(new_masks[-1].sum().item())
    ]
    agent.cfg.episode_lengths = list(agent.cfg.episode_lengths) + [
        _episode_length(target_task)
    ]
    agent.discount = torch.cat(
        [agent.discount, agent.discount[source_idx : source_idx + 1]], dim=0
    )
    _rebuild_optimizers(agent)
    return len(agent.cfg.tasks) - 1


def _pad_obs(obs: torch.Tensor, target_obs_dim: int) -> torch.Tensor:
    if obs.shape[0] == target_obs_dim:
        return obs
    if obs.shape[0] > target_obs_dim:
        return obs[:target_obs_dim]
    pad = torch.zeros(target_obs_dim - obs.shape[0], dtype=obs.dtype, device=obs.device)
    return torch.cat([obs, pad], dim=0)


def _to_td(env, obs, task_idx, action_dim, action=None, reward=None, terminated=None):
    from tensordict.tensordict import TensorDict

    obs = obs.unsqueeze(0).cpu()
    if action is None:
        action = torch.full((action_dim,), float("nan"), dtype=torch.float32)
    if reward is None:
        reward = torch.tensor(float("nan"))
    if terminated is None:
        terminated = torch.tensor(float("nan"))
    task_tensor = torch.tensor([int(task_idx)], dtype=torch.int64)
    return TensorDict(
        obs=obs,
        action=action.unsqueeze(0),
        reward=reward.unsqueeze(0),
        terminated=terminated.unsqueeze(0),
        task=task_tensor,
        batch_size=(1,),
    )


def _evaluate(agent, env, obs_dim, task_idx, episodes):
    ep_rewards, ep_successes, ep_lengths = [], [], []
    for _ in range(episodes):
        obs, done, ep_reward, t = env.reset(), False, 0.0, 0
        while not done:
            obs_pad = _pad_obs(obs, obs_dim)
            action = agent.act(obs_pad, t0=t == 0, eval_mode=True, task=task_idx)
            obs, reward, done, info = env.step(action[: env.action_space.shape[0]])
            ep_reward += float(reward)
            t += 1
        ep_rewards.append(ep_reward)
        ep_successes.append(float(info["success"]))
        ep_lengths.append(t)
    return {
        "episode_reward": float(sum(ep_rewards) / len(ep_rewards)),
        "episode_success": float(sum(ep_successes) / len(ep_successes)),
        "episode_length": float(sum(ep_lengths) / len(ep_lengths)),
    }


@hydra.main(config_name="config", config_path="../cfgs")
def off2on(cfg):
    """Offline-to-online full-model finetuning for MBDPO with a new task embedding ID."""
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    checkpoint = str(cfg.get("checkpoint", "") or "").strip()
    load_checkpoint = bool(getattr(cfg, "load_checkpoint", True))
    print(load_checkpoint)
    save_path = str(cfg.get("save_path", "") or "").strip()
    off2on_task = str(cfg.get("off2on_task", "") or "").strip()
    if load_checkpoint and not checkpoint:
        raise ValueError("Please provide `checkpoint=/path/to/mt80_19m_checkpoint.pt`.")
    if not save_path:
        raise ValueError("Please provide `save_path=/path/to/save_dir`.")
    if not off2on_task:
        raise ValueError(
            'Please provide `off2on_task`, e.g. `off2on_task="Walker Run"`.'
        )

    target_task = _canonical_task_name(off2on_task)
    source_task = SOURCE_TASK_MAP[target_task]

    cfg.task = "mt80"
    cfg.multitask = True
    cfg.task_dim = 96
    cfg.tasks = list(TASK_SET["mt80"])
    cfg.steps = int(getattr(cfg, "steps", 40_000) or 40_000)
    cfg.steps = 40_000 if cfg.steps <= 0 else cfg.steps
    cfg.eval_freq = int(getattr(cfg, "eval_freq", 5_000) or 5_000)
    cfg.eval_episodes = int(getattr(cfg, "eval_episodes", 10) or 10)
    if cfg.eval_episodes <= 0:
        raise ValueError("`eval_episodes` must be > 0")
    cfg.diffusion_steps = int(getattr(cfg, "diffusion_steps", 20) or 20)
    if cfg.diffusion_steps < 2:
        raise ValueError("`diffusion_steps` must be >= 2 for diffusion planner.")

    if load_checkpoint:
        state_dict = _load_checkpoint_state_dict(checkpoint)
        obs_dim, action_dim, action_dims = _infer_mt80_dims(state_dict, cfg.task_dim)
    else:
        obs_dim, action_dim, action_dims = _default_mt80_dims(cfg)
        print(
            colored(
                f"[Off2On] load_checkpoint=false, using default MT80 dims: obs_dim={obs_dim}, action_dim={action_dim}",
                "yellow",
                attrs=["bold"],
            )
        )
    cfg.obs = "state"
    cfg.obs_shape = {"state": (obs_dim,)}
    cfg.action_dim = action_dim
    cfg.action_dims = action_dims
    cfg.episode_lengths = [_episode_length(t) for t in cfg.tasks]
    cfg.episode_length = _episode_length(target_task)
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)

    target_env_cfg = deepcopy(cfg)
    target_env_cfg.multitask = False
    target_env_cfg.task = target_task
    env = make_env(target_env_cfg)
    eval_env = make_env(target_env_cfg)

    agent = MBDPO(cfg)
    if load_checkpoint:
        agent.load(checkpoint)
    else:
        print(
            colored(
                "[Off2On] Skipping checkpoint load (no-load baseline).",
                "yellow",
                attrs=["bold"],
            )
        )
    new_task_idx = _append_new_task_embedding(
        agent, target_task=target_task, source_task=source_task
    )
    buffer = Buffer(cfg)

    out_dir = Path(save_path) / target_task
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / f"train_{cfg.seed}.csv"
    eval_csv = out_dir / f"eval_{cfg.seed}.csv"
    log_file = out_dir / f"logs_{cfg.seed}.txt"
    model_file = out_dir / f"final_{cfg.seed}.pt"

    with open(train_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["step", "episode", "episode_reward", "episode_success", "episode_length"]
        )
    with open(eval_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["step", "episode_reward", "episode_success", "episode_length"]
        )

    print(
        colored(
            f"[Off2On] target={target_task} source={source_task} new_task_idx={new_task_idx}",
            "yellow",
            attrs=["bold"],
        )
    )
    print(
        colored(
            f'[Off2On] contrastive_beta={float(getattr(cfg, "contrastive_beta", 0.01))} diffusion_steps={cfg.diffusion_steps} eval_episodes={cfg.eval_episodes}',
            "yellow",
            attrs=["bold"],
        )
    )
    print(colored(f"[Off2On] output_dir={out_dir}", "yellow", attrs=["bold"]))

    if cfg.eval_freq > 0:
        init_eval = _evaluate(agent, eval_env, obs_dim, new_task_idx, cfg.eval_episodes)
        with open(eval_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    0,
                    init_eval["episode_reward"],
                    init_eval["episode_success"],
                    init_eval["episode_length"],
                ]
            )
        with open(log_file, "a") as f:
            f.write(
                f'EVAL step=0 reward={init_eval["episode_reward"]:.4f} success={init_eval["episode_success"]:.4f} len={init_eval["episode_length"]:.2f}\n'
            )

    step, episode = 0, 0
    obs = env.reset()
    obs_pad = _pad_obs(obs, obs_dim)
    tds = [_to_td(env, obs_pad, task_idx=new_task_idx, action_dim=cfg.action_dim)]

    while step <= cfg.steps:
        if cfg.eval_freq > 0 and step > 0 and step % cfg.eval_freq == 0:
            eval_metrics = _evaluate(
                agent, eval_env, obs_dim, new_task_idx, cfg.eval_episodes
            )
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        step,
                        eval_metrics["episode_reward"],
                        eval_metrics["episode_success"],
                        eval_metrics["episode_length"],
                    ]
                )
            with open(log_file, "a") as f:
                f.write(
                    f'EVAL step={step} reward={eval_metrics["episode_reward"]:.4f} success={eval_metrics["episode_success"]:.4f} len={eval_metrics["episode_length"]:.2f}\n'
                )

        use_random_exploration = (not load_checkpoint) and (step <= cfg.seed_steps)
        if use_random_exploration:
            action = env.rand_act()
        else:
            action = agent.act(obs_pad, t0=len(tds) == 1, task=new_task_idx)
        if action.shape[0] < cfg.action_dim:
            action = torch.cat(
                [
                    action,
                    torch.zeros(cfg.action_dim - action.shape[0], dtype=action.dtype),
                ],
                dim=0,
            )

        next_obs, reward, done, info = env.step(action[: env.action_space.shape[0]])
        next_obs_pad = _pad_obs(next_obs, obs_dim)
        tds.append(
            _to_td(
                env,
                next_obs_pad,
                task_idx=new_task_idx,
                action_dim=cfg.action_dim,
                action=action,
                reward=reward,
                terminated=info["terminated"],
            )
        )

        if load_checkpoint:
            if buffer.num_eps > 0:
                agent.update(buffer)
        else:
            if step >= cfg.seed_steps:
                num_updates = cfg.seed_steps if step == cfg.seed_steps else 1
                for _ in range(num_updates):
                    agent.update(buffer)

        if done:
            episode_td = torch.cat(tds)
            buffer.add(episode_td)
            episode_reward = float(torch.stack([x["reward"] for x in tds[1:]]).sum())
            episode += 1
            with open(train_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step, episode, episode_reward, float(info["success"]), len(tds)]
                )
            with open(log_file, "a") as f:
                f.write(
                    f'TRAIN step={step} episode={episode} reward={episode_reward:.4f} success={float(info["success"]):.4f} len={len(tds)}\n'
                )

            obs = env.reset()
            obs_pad = _pad_obs(obs, obs_dim)
            tds = [
                _to_td(env, obs_pad, task_idx=new_task_idx, action_dim=cfg.action_dim)
            ]
        else:
            obs = next_obs
            obs_pad = next_obs_pad

        step += 1

    agent.save(model_file)
    print(colored(f"[Off2On] Saved model to {model_file}", "green", attrs=["bold"]))
    print(colored("[Off2On] Finished successfully.", "green", attrs=["bold"]))


if __name__ == "__main__":
    off2on()
