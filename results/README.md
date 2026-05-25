# Task Sets and Result Domains (`results/`)

This document lists **every task** in each task set/domain used by this repository, and how to pass them to training/evaluation scripts.

## How to use task names

You can pass a single task name (e.g., `task=dog-run`) or a task-set alias (e.g., `task=mt80`) to:

- `scripts/evaluate.py`
- `scripts/train.py`
- `scripts/offline_to_online.py`
- `scripts/online_parallel_train.py`

Multi-task training/evaluation is specified by:

- `task=mt80` for the 80-task set.
- `task=mt30` for the 30-task set.

---

## `mt30` (30 tasks)

```text
walker-stand
walker-walk
walker-run
cheetah-run
reacher-easy
reacher-hard
acrobot-swingup
pendulum-swingup
cartpole-balance
cartpole-balance-sparse
cartpole-swingup
cartpole-swingup-sparse
cup-catch
finger-spin
finger-turn-easy
finger-turn-hard
fish-swim
hopper-stand
hopper-hop
walker-walk-backwards
walker-run-backwards
cheetah-run-backwards
cheetah-run-front
cheetah-run-back
cheetah-jump
hopper-hop-backwards
reacher-three-easy
reacher-three-hard
cup-spin
pendulum-spin
```

## `mt80` (80 tasks)

> `mt80 = mt30 + 50 MetaWorld tasks`

```text
walker-stand
walker-walk
walker-run
cheetah-run
reacher-easy
reacher-hard
acrobot-swingup
pendulum-swingup
cartpole-balance
cartpole-balance-sparse
cartpole-swingup
cartpole-swingup-sparse
cup-catch
finger-spin
finger-turn-easy
finger-turn-hard
fish-swim
hopper-stand
hopper-hop
walker-walk-backwards
walker-run-backwards
cheetah-run-backwards
cheetah-run-front
cheetah-run-back
cheetah-jump
hopper-hop-backwards
reacher-three-easy
reacher-three-hard
cup-spin
pendulum-spin
mw-assembly
mw-basketball
mw-button-press-topdown
mw-button-press-topdown-wall
mw-button-press
mw-button-press-wall
mw-coffee-button
mw-coffee-pull
mw-coffee-push
mw-dial-turn
mw-disassemble
mw-door-open
mw-door-close
mw-drawer-close
mw-drawer-open
mw-faucet-open
mw-faucet-close
mw-hammer
mw-handle-press-side
mw-handle-press
mw-handle-pull-side
mw-handle-pull
mw-lever-pull
mw-peg-insert-side
mw-peg-unplug-side
mw-pick-out-of-hole
mw-pick-place
mw-pick-place-wall
mw-plate-slide
mw-plate-slide-side
mw-plate-slide-back
mw-plate-slide-back-side
mw-push-back
mw-push
mw-push-wall
mw-reach
mw-reach-wall
mw-shelf-place
mw-soccer
mw-stick-push
mw-stick-pull
mw-sweep-into
mw-sweep
mw-window-open
mw-window-close
mw-bin-picking
mw-box-close
mw-door-lock
mw-door-unlock
mw-hand-insert
```

---

## DMControl (39 tasks)

```text
walker-stand
walker-walk
walker-run
cheetah-run
reacher-easy
reacher-hard
acrobot-swingup
pendulum-swingup
cartpole-balance
cartpole-balance-sparse
cartpole-swingup
cartpole-swingup-sparse
cup-catch
finger-spin
finger-turn-easy
finger-turn-hard
fish-swim
hopper-stand
hopper-hop
walker-walk-backwards
walker-run-backwards
cheetah-run-backwards
cheetah-run-front
cheetah-run-back
cheetah-jump
hopper-hop-backwards
reacher-three-easy
reacher-three-hard
cup-spin
pendulum-spin
quadruped-walk
quadruped-run
humanoid-walk
humanoid-run
humanoid-stand
dog-walk
dog-run
dog-stand
dog-trot
```

## MetaWorld (50 tasks)

```text
mw-assembly
mw-basketball
mw-button-press-topdown
mw-button-press-topdown-wall
mw-button-press
mw-button-press-wall
mw-coffee-button
mw-coffee-pull
mw-coffee-push
mw-dial-turn
mw-disassemble
mw-door-open
mw-door-close
mw-drawer-close
mw-drawer-open
mw-faucet-open
mw-faucet-close
mw-hammer
mw-handle-press-side
mw-handle-press
mw-handle-pull-side
mw-handle-pull
mw-lever-pull
mw-peg-insert-side
mw-peg-unplug-side
mw-pick-out-of-hole
mw-pick-place
mw-pick-place-wall
mw-plate-slide
mw-plate-slide-side
mw-plate-slide-back
mw-plate-slide-back-side
mw-push-back
mw-push
mw-push-wall
mw-reach
mw-reach-wall
mw-shelf-place
mw-soccer
mw-stick-push
mw-stick-pull
mw-sweep-into
mw-sweep
mw-window-open
mw-window-close
mw-bin-picking
mw-box-close
mw-door-lock
mw-door-unlock
mw-hand-insert
```

## ManiSkill2 (5 tasks)

```text
lift-cube
pick-cube
stack-cube
pick-ycb
turn-faucet
```

## MyoSuite (10 tasks)

```text
myo-reach
myo-reach-hard
myo-pose
myo-pose-hard
myo-obj-hold
myo-obj-hold-hard
myo-key-turn
myo-key-turn-hard
myo-pen-twirl
myo-pen-twirl-hard
```

## Locomotion (7 tasks)

```text
dog-walk
dog-run
dog-stand
dog-trot
humanoid-walk
humanoid-run
humanoid-stand
```

## Visual RL (10 tasks)

```text
acrobot-swingup
cheetah-run
finger-spin
finger-turn-easy
finger-turn-hard
quadruped-walk
reacher-easy
reacher-hard
walker-run
walker-walk
```

---

## Notes

If you want to run Visual online RL task in the DMControl tasks, please use argument `obs=rgb` in the scripts.