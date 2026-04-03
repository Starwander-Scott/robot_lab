# MuJoCo Gamepad 控制说明

本文档只说明如何用手柄操控本仓库的 MuJoCo Go2 验证脚本。

## 1. 脚本位置

- 手柄版验证脚本：`scripts/mujoco/verify_go2_policy_gamepad.py`
- 相关参考实现：`/home/oepr/Desktop/github_proj/unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py`

## 2. 运行前准备

需要安装：

- `mujoco`
- `torch`
- `pygame`

安装方式：

```bash
/home/oepr/robot_lab/.venv/bin/python -m pip install torch mujoco pygame
```

推荐使用：

- `assets/go2/scene.xml`
- 已导出的 TorchScript policy 文件

## 3. 启动方式

```bash
/home/oepr/robot_lab/.venv/bin/python scripts/mujoco/verify_go2_policy_gamepad.py \
  --mjcf /home/oepr/robot_lab/assets/go2/scene.xml \
  --policy /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt \
  --control-mode pd \
  --render \
  --gamepad-type xbox
```

常用参数：

- `--gamepad-id`：选择第几个手柄设备
- `--gamepad-type {xbox,switch}`：选择手柄布局
- `--max-vx`：前后速度缩放
- `--max-vy`：横移速度缩放
- `--max-wz`：转向角速度缩放
- `--deadband`：摇杆死区

## 4. 手柄映射

脚本把摇杆输出映射到 `cmd=[vx, vy, wz]`。

### Xbox 布局

- `LX=0`
- `LY=1`
- `RX=3`
- `RY=4`
- `LT=2`
- `RT=5`

### Switch 布局

- `LX=0`
- `LY=1`
- `RX=2`
- `RY=3`
- `LT=5`
- `RT=4`

## 5. 控制逻辑

- 左摇杆 `Y` 控制 `vx`
- 左摇杆 `X` 控制 `vy`
- 右摇杆 `X` 控制 `wz`
- `A` 键或 `LT` 会把 `cmd` 清零

如果你想换成别的按键逻辑，只需要改：

- `scripts/mujoco/verify_go2_policy_gamepad.py`

## 6. 和策略的关系

这个模式不改策略网络本身，只是把实时手柄输入写进观测里的速度命令部分。

对 Go2 来说，观测里命令向量的位置在：

- `base_ang_vel`
- `projected_gravity`
- `cmd`
- `joint_pos`
- `joint_vel`
- `last_action`

因此手柄控制的核心不是改 `policy.pt`，而是改 `cmd` 的来源。

## 7. 常见问题

### 7.1 没识别到手柄

- 检查系统是否能看到 joystick 设备
- 尝试切换 `--gamepad-id`
- 确认已安装 `pygame`

### 7.2 方向反了

如果你的手柄布局和 Xbox/Switch 不一致，改 `--gamepad-type`。
如果只是单个轴方向反了，改脚本里 `_update_command_from_gamepad()` 的符号。

### 7.3 机器人抖动

优先检查：

- `--control-mode pd`
- `--kp 25 --kd 0.5`
- `control_dt=0.02`
- 使用 `assets/go2/scene.xml`

