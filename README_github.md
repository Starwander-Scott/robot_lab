# 🐾 Unitree Go2 Sim2Sim: IsaacLab to MuJoCo

A robust and minimalist pipeline for transferring Unitree Go2 reinforcement learning policies trained in **IsaacLab** (via RSL-RL) to the **MuJoCo** physics engine for high-fidelity Sim2Sim verification. 

This repository serves as a crucial stepping stone before real-world deployment (Sim2Real), ensuring your policy is stable under strict physical constraints.

## ✨ Key Features
* **Seamless Policy Export**: Extracts the pure Actor MLP from bloated RSL-RL checkpoints and compiles it into a standalone `TorchScript` (`.pt`) model for easy deployment.
* **Precise Observation Mapping**: Perfectly perfectly aligns MuJoCo's physical state (projected gravity, local velocities, joint positions) with the 45-dimensional observation space expected by the IsaacLab-trained model.
* **Stable PD Control**: Implements customized PD torque controllers matching training configurations (`kp=25`, `kd=0.5`).
* **Zero Start-up Explosions**: Fixes the notorious "flailing/crashing" bug caused by zero-state initialization by forcing nominal joint presets before simulating the first frame.
* **Real-time Rendering**: Includes wall-time synchronization for smooth UI viewing without fast-forwarding.

## 📂 Core Scripts
* `scripts/mujoco/export_go2_jit.py` : Script to extract the neural network weights and export the JIT Actor.
* `scripts/mujoco/verify_go2_policy.py` : The main MuJoCo simulator runner.
* `assets/go2/scene.xml` : The customized MuJoCo stage with collision floors and skyboxes.

## 🚀 Quick Start

### 1. Export the Policy
Extract the trained model from your IsaacLab/RSL-RL log directory:
```bash
python scripts/mujoco/export_go2_jit.py \
    --checkpoint logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/model_4300.pt \
    --output policy_mujoco.pt
```

### 2. Run the Sim2Sim Verification
Launch the MuJoCo viewer and let the Go2 walk. You can control the target velocity using the command-line arguments:
```bash
# Walk forward at 0.5 m/s
python scripts/mujoco/verify_go2_policy.py \
    --mjcf assets/go2/scene.xml \
    --policy policy_mujoco.pt \
    --duration 20 \
    --control-mode pd \
    --cmd-vx 0.5 \
    --render

# Walk sideways (Crab walk) at 0.4 m/s
python scripts/mujoco/verify_go2_policy.py \
    --mjcf assets/go2/scene.xml \
    --policy policy_mujoco.pt \
    --cmd-vy 0.4 \
    --render
```

## 🧠 Tutorial
If you are new to Sim2Sim transfer or experiencing robot instability in simulators, please check out our detailed guide: [tutorial.md](tutorial.md) (in Chinese) for a step-by-step breakdown of how the observation alignment and PD controllers are built.