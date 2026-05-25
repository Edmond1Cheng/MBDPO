# Conda Environment Notes

This document provides key setup notes for `conda_envs/*.yml` across task domains. Create the environment from the YAML first, then verify domain-specific runtime variables and dependencies listed below.

## General recommendations (all domains)

1. **Use separate environments exactly as named in YAML** to avoid cross-domain contamination:
   - `mbdpo-mt80`
   - `mbdpo-ms2`
   - `mbdpo-myo`
2. **Install in a fresh environment** whenever possible. Avoid in-place upgrades of core packages (for example: `torch`, `mujoco`, `gym`) in active experiment envs.
3. On headless servers/containers, prefer **EGL/headless rendering** over default GLX to avoid renderer initialization failures.
4. For multi-process training, if you see CPU oversubscription or degraded throughput, cap BLAS/OMP threads (see ManiSkill2 section).
5. Put required `export` commands in startup scripts so behavior is consistent across `tmux`, `nohup`, and schedulers (for example Slurm).

---

## Domain: MT80 (`mbdpo-mt80`)

### 1) MuJoCo / OpenGL runtime variables (required)

Set these before training:

```bash
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin
```

Notes:
- `MUJOCO_GL=egl`: preferred backend for headless machines.
- `PYOPENGL_PLATFORM=egl`: keeps the OpenGL platform aligned with MuJoCo backend.
- `LD_LIBRARY_PATH`: ensures MuJoCo 2.1.0 runtime libraries are discoverable.

### 2) Meta-World compatibility constraints (important)

- Meta-World has a fragile dependency chain and typically requires:
  - **MuJoCo 2.1.0**
  - **gym==0.21.0**
- `gym==0.21.0` is increasingly difficult to install with modern packaging toolchains.
- If installation fails, recommended checks:
  1. Use the provided `mbdpo-mt80.yml` as-is.
  2. Avoid arbitrary upgrades of `pip` / `setuptools`.
  3. Keep Python major version aligned with YAML (3.9).

### 3) `mjkey.txt` setup (required)

For Meta-World + `mujoco-py` compatibility paths, a license key may still be required. Put the following content in `~/.mujoco/mjkey.txt` (courtesy of Google DeepMind):

```text
MuJoCo Pro Individual license activation key, number 7777, type 6.

Issued to Everyone.

Expires October 18, 2031.

Do not modify this file. Its entire content, including the
plain text section, is used by the activation manager.

9aaedeefb37011a8a52361c736643665c7f60e796ff8ff70bb3f7a1d78e9a605
0453a3c853e4aa416e712d7e80cf799c6314ee5480ec6bd0f1ab51d1bb3c768f
8c06e7e572f411ecb25c3d6ef82cc20b00f672db88e6001b3dfdd3ab79e6c480
185d681811cfdaff640fb63295e391b05374edba90dd54cc1e162a9d99b82a8b
ea3e87f2c67d08006c53daac2e563269cdb286838b168a2071c48c29fedfbea2
5effe96fe3cb05e85fb8af2d3851f385618ef8cdac42876831f095e052bd18c9
5dce57ff9c83670aad77e5a1f41444bec45e30e4e827f7bf9799b29f2c934e23
dcf6d3c3ee9c8dd2ed057317100cd21b4abbbf652d02bf72c3d322e0c55dcc24
```

Additional suggestion:
- Also set `export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210`
- Verify `~/.mujoco/mjkey.txt` exists and is readable.

---

## Domain: ManiSkill2 (`mbdpo-ms2`)

### 1) Asset path (required)

Download the required task assets and export:

```bash
export MS2_ASSET_DIR=/XXX/XXX/XXXX
```

Without this variable, common failures include missing task/model/mesh files.

### 2) Vulkan + headless rendering (required)

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
```

Notes:
- `VK_ICD_FILENAMES`: explicitly points to NVIDIA Vulkan ICD.
- `NVIDIA_DRIVER_CAPABILITIES`: often required in containers to enable graphics capability.
- `XDG_RUNTIME_DIR`: prevents runtime dir permission issues during SAPIEN/Vulkan initialization.

### 3) Thread balancing for parallel online training (recommended)

For parallel online training, you can set:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

This usually reduces oversubscription and keeps load more balanced. Tune based on physical CPU cores.

### 4) HuggingFace tokenizer parallelism (recommended)

Add the missing setting:

```bash
export TOKENIZERS_PARALLELISM=false
```

Purpose:
- Avoid duplicated parallelism and warnings in multi-process/multi-thread setups.
- Can improve stability and log cleanliness.

---

## Domain: MyoSuite (`mbdpo-myo`)

### 1) Rendering backend and display mode

For headless remote training (if rendering is involved), EGL is recommended:

```bash
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
```

If you need local desktop visualization, you can switch to `glfw`/default backend as needed, but avoid mixing with batch training settings.

### 2) MuJoCo version consistency

- `mbdpo-myo.yml` uses a newer `mujoco` stack (not legacy `mujoco-py`).
- Do not copy MT80-specific MuJoCo 2.1.0 / `mujoco-py` constraints into MyoSuite env, or ABI/API conflicts may occur.

### 3) Training stability

- If CPU contention or throughput jitter appears, apply the same thread-cap guidance used for ManiSkill2.
- Record critical env variables in experiment logs (or `wandb.config`) for reproducibility.

---

## Recommended practice: domain-specific env bootstrap scripts

Maintain one startup script per domain, for example:
- `env_mt80.sh`
- `env_ms2.sh`
- `env_myo.sh`

Then source it before launch:

```bash
source scripts/env_mt80.sh  # or your corresponding domain script
```

This significantly reduces machine-to-machine configuration drift.
