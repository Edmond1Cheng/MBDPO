# Conda Environment Notes

This branch is JAX-only. Use:

```bash
conda env create -f conda_envs/mbdpo-jax.yml
conda activate mbdpo-jax
```

The verified felis environment uses `jax[cuda12]==0.7.1` with cuDNN `>=9.8`.
The YAML pins `nvidia-cudnn-cu12==9.8.0.87` because earlier cuDNN builds failed
JAX GPU initialization on that machine.

For headless rendering, keep:

```bash
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
```
