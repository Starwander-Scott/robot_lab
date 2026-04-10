# MuJoCo Go2 验证脚本使用说明

本文档说明如何使用本仓库新增的 MuJoCo 脚本进行 Go2 策略验证。

## 1. 文件位置

- 导出脚本：scripts/mujoco/export_go2_jit.py
- 验证脚本：scripts/mujoco/verify_go2_policy.py
- 手柄验证说明：scripts/mujoco/gamepad_control.md
- 手柄验证脚本：scripts/mujoco/verify_go2_policy_gamepad.py
- Go2 模型（推荐）：assets/go2/scene.xml
- Go2 资产目录：assets/go2/assets/

## 2. 环境准备

使用仓库的虚拟环境：

- Python：/home/oepr/robot_lab/.venv/bin/python

确保已安装依赖：

- torch
- mujoco
- pygame（手柄控制时需要）

如需安装：

- /home/oepr/robot_lab/.venv/bin/python -m pip install torch mujoco
- /home/oepr/robot_lab/.venv/bin/python -m pip install pygame

> 注意：本项目不再依赖 rsl_rl 来导出 TorchScript。

## 3. 导出 TorchScript 策略

将 RSL-RL 的 checkpoint 导出为 TorchScript actor：

- 运行：
  /home/oepr/robot_lab/.venv/bin/python scripts/mujoco/export_go2_jit.py \
    --checkpoint /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/model_4300.pt \
    --output /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt

导出成功后会生成：
- logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt

## 4. 运行 MuJoCo 验证

推荐使用 assets/go2/scene.xml（包含地面与正确的 meshdir）：

- 运行：
  /home/oepr/robot_lab/.venv/bin/python scripts/mujoco/verify_go2_policy.py \
    --mjcf /home/oepr/robot_lab/assets/go2/scene.xml \
    --policy /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt \
    --duration 20 \
    --control-mode pd

可选参数：

- --render：开启可视化
- --cmd-vx / --cmd-vy / --cmd-wz：速度指令
- --control-dt：控制周期（默认 0.02）
- --sim-dt：仿真步长（默认使用 XML 内设置）
- --kp / --kd：PD 参数（Go2 训练默认：kp=25, kd=0.5）
- --spawn-x / --spawn-y / --spawn-z：覆盖机器人出生点
- --spawn-yaw-deg：覆盖机器人初始朝向
- --base-body：底座 body 名称（默认 base_link）

家庭场景说明：

- `scene_home.xml`
- `scene_home_300.xml`
- `scene_home_300_nav.xml`

这几个家庭场景会自动把 Go2 出生点放到室内开阔区域，避免默认原点出生导致贴墙卡住。
当前默认出生点约为：

- `x=5.2`
- `y=1.4`
- `z=0.45`
- `yaw=0`

示例（带可视化）：

- 运行：
  /home/oepr/robot_lab/.venv/bin/python scripts/mujoco/verify_go2_policy.py \
    --mjcf /home/oepr/robot_lab/assets/go2/scene.xml \
    --policy /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt \
  --duration 20 \
  --control-mode pd \
  --render

示例（300 平米家庭导航场景）：

- 运行：
  /home/oepr/robot_lab/.venv/bin/python scripts/mujoco/verify_go2_policy.py \
    --mjcf /home/oepr/robot_lab/assets/go2/scene_home_300_nav.xml \
    --policy /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt \
    --duration 45 \
    --control-mode pd \
    --render

示例（手动指定出生点）：

- 运行：
  /home/oepr/robot_lab/.venv/bin/python scripts/mujoco/verify_go2_policy.py \
    --mjcf /home/oepr/robot_lab/assets/go2/scene_home_300_nav.xml \
    --policy /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt \
    --duration 45 \
    --control-mode pd \
    --spawn-x 5.2 \
    --spawn-y 1.4 \
    --spawn-yaw-deg 0 \
    --render

## 4.1 使用手柄控制

独立说明见：

- `scripts/mujoco/gamepad_control.md`

## 5. 常见问题

### 5.1 mesh 读取失败

请确认 MJCF 的 meshdir 指向：
- assets/go2/assets

推荐直接使用：
- assets/go2/scene.xml

### 5.2 关节名称不匹配

脚本默认使用 Go2 关节列表（与 unitree_go2_rough 一致）。如果 MJCF 中关节名不同，需要在：
- scripts/mujoco/verify_go2_policy.py
修改 GO2_JOINT_NAMES。

### 5.3 观测维度不匹配

当前脚本构造的观测为 Go2 rough 默认配置：
- base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action

如需对齐其他配置，请在：
- scripts/mujoco/verify_go2_policy.py
调整 _build_obs。

### 5.4 机器人乱动/抖动（最常见）

请逐项对齐以下关键项：

1) PD 参数需与训练一致
- Go2 训练默认：kp=25, kd=0.5
- 若使用更大 kp/kd，常见现象是抖动或“乱动”

2) 初始关节角需要一致
- Isaac 训练初始：hip=0, thigh=0.8, calf=-1.5
- 当前 assets/go2/go2.xml 的 keyframe 为 0.9 / -1.8（不一致）
- 建议将 assets/go2/go2.xml 的 <keyframe> 里关节角改为 0.8 / -1.5

3) 控制周期需与训练一致
- 训练配置：sim.dt=0.005，decimation=4
- 因此 control_dt 应为 0.02（脚本默认值已匹配）

4) 使用带地面的场景文件
- 必须使用 assets/go2/scene.xml
- 直接使用 go2.xml 会导致空中下落

5) 观测顺序必须与训练一致
- Go2 rough 观测顺序：base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, last_action
- 脚本已按该顺序构造，若改动请保持一致
