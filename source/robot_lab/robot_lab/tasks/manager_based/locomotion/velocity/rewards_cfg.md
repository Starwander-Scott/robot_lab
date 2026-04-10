# RewardsCfg 详尽说明（velocity_env_cfg.py）

> 位置：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` 中 `RewardsCfg`

## 1. 作用概览
`RewardsCfg` 定义 **奖励函数的组成与权重**。在 Manager-Based 结构中，每个奖励项由 `RewTerm` 描述，主要包含：
- `func`: 对应奖励计算函数（在 `mdp` 模块中实现）
- `weight`: 权重系数（正为奖励，负为惩罚）
- `params`: 需要的实体或参数配置

目前该配置 **大多数权重为 0.0**，表示占位或暂时禁用。

---

## 2. 奖励项逐一解析

### 2.1 General
- `is_terminated`: 终止惩罚或终止标记。

```python
is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
```

---

### 2.2 Root penalties（根部相关惩罚）

- `lin_vel_z_l2`: 惩罚竖直速度（跳跃或离地过快）。
- `ang_vel_xy_l2`: 惩罚 x/y 角速度（翻滚/侧翻）。
- `flat_orientation_l2`: 惩罚姿态偏离水平。
- `base_height_l2`: 惩罚 base 高度偏离目标。
- `body_lin_acc_l2`: 惩罚线加速度，抑制震荡。

```python
lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
base_height_l2 = RewTerm(
    func=mdp.base_height_l2,
    weight=0.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0},
)
body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="")})
```

---

### 2.3 Joint penalties（关节惩罚）

- `joint_torques_l2`: 惩罚关节力矩过大。  
- `joint_vel_l2`: 惩罚关节速度过大。  
- `joint_acc_l2`: 惩罚关节加速度过大。  
- `joint_pos_limits`: 惩罚关节越界。  
- `joint_vel_limits`: 惩罚关节速度超限。  
- `joint_power`: 惩罚关节功率消耗。  

```python
joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
joint_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0})
joint_power = RewTerm(func=mdp.joint_power, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
```

- `create_joint_deviation_l1_rewterm`: 工具函数，用于按需创建关节偏差惩罚项（L1）。

---

### 2.4 Stand still & joint position penalties

- `stand_still`: 在低速度指令下惩罚不稳定动作。  
- `joint_pos_penalty`: 约束关节偏移在合理范围内。

```python
stand_still = RewTerm(
    func=mdp.stand_still,
    weight=0.0,
    params={"command_name": "base_velocity", "command_threshold": 0.1, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
)

joint_pos_penalty = RewTerm(
    func=mdp.joint_pos_penalty,
    weight=0.0,
    params={
        "command_name": "base_velocity",
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stand_still_scale": 5.0,
        "velocity_threshold": 0.5,
        "command_threshold": 0.1,
    },
)
```

---

### 2.5 Wheel related
- `wheel_vel_penalty`: 针对轮式/腿部轮式相关约束（本项目中可能用于特定模型）。

---

### 2.6 Symmetry & sync
- `joint_mirror`: 关节对称奖励。  
- `action_mirror`: 动作对称奖励。  
- `action_sync`: 同步多个关节组（如髋/大腿/小腿同步）。

---

### 2.7 Action penalties
- `applied_torque_limits`: 力矩限制惩罚。  
- `action_rate_l2`: 动作变化率惩罚（平滑动作）。

---

### 2.8 Contact sensor
- `undesired_contacts`: 非法接触惩罚。  
- `contact_forces`: 接触力惩罚。

---

### 2.9 Velocity-tracking rewards
- `track_lin_vel_xy_exp`: 速度跟踪奖励（指数形式）。  
- `track_ang_vel_z_exp`: 角速度跟踪奖励（指数形式）。

---

### 2.10 Feet-related
- `feet_air_time`: 空中时间奖励。  
- `feet_air_time_variance`: 空中时间方差惩罚。  
- `feet_gait`: 步态奖励。  
- `feet_contact`: 期望接触脚数量。  
- `feet_contact_without_cmd`: 无速度指令时的接触约束。  
- `feet_stumble`: 绊倒惩罚。  
- `feet_slide`: 足端滑动惩罚。  
- `feet_height`: 足端高度奖励。  
- `feet_height_body`: 足端相对机体高度奖励。  
- `feet_distance_y_exp`: 站距奖励（横向）。

---

### 2.11 Others
- `upward`: 向上朝向奖励（保持机体竖直）。

---

## 3. 训练与调参建议

- **若机器人抖动**：提高 `action_rate_l2` 或 `joint_vel_l2` 惩罚。  
- **若打滑严重**：提高 `feet_slide` 惩罚或调高摩擦随机下限。  
- **若步态不稳定**：适当增加 `feet_gait` 或 `feet_air_time` 权重。  
- **若速度跟踪差**：提高 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`。  

---

## 4. 小结
`RewardsCfg` 通过丰富的奖励项组合，实现对步态稳定、能耗、平衡和速度跟踪的综合约束。虽然当前权重大多为 0.0，但该结构便于后续按需求逐项启用和调参。