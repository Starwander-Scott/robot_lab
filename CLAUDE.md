# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**robot_lab** is a reinforcement learning extension library for robots, built on top of [IsaacLab](https://isaac-sim.github.io/IsaacLab) (which runs on NVIDIA Isaac Sim). It provides training environments for quadruped, wheeled, and humanoid robots using GPU-accelerated RL.

**Primary RL framework**: RSL-RL (PPO). CusRL and SKRL are experimental.

## Installation

```bash
# Install into the active Isaac Lab Python environment
python -m pip install -e source/robot_lab
```

## Common Commands

### Training & Inference

```bash
# Train (headless)
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 --headless

# Resume from checkpoint
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 --headless \
  --resume --load_run <run_folder> --checkpoint /PATH/TO/model.pt

# Multi-GPU training
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 --headless --distributed

# Play/inference
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 --num_envs=32

# Record video during play
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 --video --video_length=200
```

### Smoke Testing / Debugging

```bash
# List all registered environments (also verifies installation)
python scripts/tools/list_envs.py

# Test environment with zero actions (no learning required)
python scripts/tools/zero_agent.py --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0

# Test with random actions
python scripts/tools/random_agent.py --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0

# Monitor training
tensorboard --logdir=logs
```

### Linting & Formatting

```bash
# Run all pre-commit checks (Ruff lint + format, codespell, YAML validation)
pre-commit run --all-files
```

Code style: Python 3.10+, 120-character line limit (Ruff), Google docstring style.

## Architecture

### Environment Naming Convention

Environments follow: `RobotLab-Isaac-<Task>-<Terrain>-<Robot>-v<X>`

Example: `RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0`
- Task: velocity locomotion control
- Terrain: rough (procedurally generated)
- Robot: Unitree A1 quadruped

### Key Directory Layout

```
source/robot_lab/robot_lab/
├── assets/              # Robot asset definitions (unitree.py, anybotics.py, etc.)
│                        # Each file defines robot morphology + references mesh files
├── tasks/
│   ├── manager_based/   # Primary task type
│   │   ├── locomotion/velocity/
│   │   │   ├── velocity_env_cfg.py      # Base task config (all robots inherit from this)
│   │   │   ├── mdp/                     # MDP components (observations, rewards, actions...)
│   │   │   └── config/<robot_family>/<robot>/
│   │   │       ├── __init__.py          # gym.register() calls
│   │   │       ├── flat_env_cfg.py
│   │   │       ├── rough_env_cfg.py
│   │   │       └── agents/rsl_rl_ppo_cfg.py
│   │   └── beyondmimic/ # Motion imitation tasks
│   └── direct/          # Direct RL environments (e.g., AMP for Unitree G1)
```

### Environment Configuration Hierarchy

```
IsaacLab ManagerBasedRLEnv (base class)
    ↑
VelocityEnvCfg (velocity_env_cfg.py) — base task config
    ↑
Robot-specific EnvCfg (e.g., UnitreeA1RoughEnvCfg)
    ↑
Registered Gym environment (via gym.register())
```

Configs use IsaacLab's `@configclass` decorator (dataclass-based inheritance). They are effectively immutable at runtime.

### MDP Components (in `mdp/`)

Each task's `mdp/` directory contains modular components:
- `observations.py` — proprioceptive + exteroceptive sensor readings
- `rewards.py` — reward terms (velocity tracking, regularization, etc.)
- `commands.py` — velocity commands or motion references
- `events.py` — resets and domain randomization
- `terminations.py` — episode end conditions
- `curriculums.py` — progressive difficulty scaling

### Training Data Flow

```
train.py → load gym env by name → instantiate ManagerBasedRLEnv
    → load scene (terrain + robot + sensors + managers)
    → gym loop: reset() → step(action) → compute rewards → next obs
    → RSL-RL PPO agent: obs → action
    → logs/checkpoints saved to logs/
```

### Adding a New Robot

1. Define asset in `source/robot_lab/robot_lab/assets/<name>.py`
2. Place mesh/URDF in `source/robot_lab/data/Robots/<manufacturer>/`
3. Create config directory:
   ```
   tasks/manager_based/locomotion/velocity/config/<family>/<robot>/
   ├── __init__.py          # gym.register() calls
   ├── flat_env_cfg.py
   ├── rough_env_cfg.py
   └── agents/rsl_rl_ppo_cfg.py
   ```
4. Verify with `python scripts/tools/list_envs.py` and `zero_agent.py`

## There Are No Unit Tests

There is no `tests/` directory. Validation is done via smoke testing with `zero_agent.py`/`random_agent.py` and short training runs with `--num_envs=4 --headless`.
