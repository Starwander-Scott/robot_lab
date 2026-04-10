# EventCfg 详尽说明（velocity_env_cfg.py）

> 位置：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` 中 `EventCfg`

## 1. 作用概览
`EventCfg` 用于配置 **环境事件（events）**，即在仿真生命周期的不同阶段对机器人/环境施加 **随机化** 或 **扰动**。其目的是提升鲁棒性与泛化能力（domain randomization）。

事件按触发时机（`mode`）分为三类：
- `startup`：环境创建/初始化时执行一次。
- `reset`：每次 episode reset 时执行。
- `interval`：按时间间隔周期性执行。

在本配置中，`EventCfg` 主要包含：
- 物理材质随机化（摩擦系数）
- 质量/质心随机化
- 外力/冲击扰动
- 关节重置噪声
- 控制器增益随机化
- 根状态随机化
- 定时推挤扰动

---

## 2. 事件项逐一解析

### 2.1 `randomize_rigid_body_material`
**作用**：随机化刚体材质摩擦与回弹系数。  
**触发**：`startup`

```python
randomize_rigid_body_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.1, 1.0),
        "dynamic_friction_range": (0.1, 0.8),
        "restitution_range": (0.0, 0.5),
        "num_buckets": 64,
    },
)
```

- `static_friction_range` / `dynamic_friction_range`：静/动摩擦系数范围。  
- `restitution_range`：弹性系数范围。  
- `num_buckets`：离散化桶数（降低随机化噪声抖动）。  
- `asset_cfg`: 目标对象，这里是所有机器人刚体。

**影响**：提高策略对不同地面摩擦的适应能力。

---

### 2.2 `randomize_rigid_body_mass_base`
**作用**：随机化机器人 base 的质量（加法模式）。  
**触发**：`startup`

```python
randomize_rigid_body_mass_base = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=""),
        "mass_distribution_params": (-1.0, 3.0),
        "operation": "add",
        "recompute_inertia": True,
    },
)
```

- `body_names=""` 通常指 base/body root。  
- `mass_distribution_params`：随机加在 base 上的质量范围。  
- `operation="add"`：加法模式。  
- `recompute_inertia=True`：质量变化后重新计算惯量。

**影响**：训练对载荷变化、机体质量误差更鲁棒。

---

### 2.3 `randomize_rigid_body_mass_others`
**作用**：随机化机器人其余刚体质量（缩放模式）。  
**触发**：`startup`

```python
randomize_rigid_body_mass_others = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "mass_distribution_params": (0.7, 1.3),
        "operation": "scale",
        "recompute_inertia": True,
    },
)
```

- `operation="scale"`：比例缩放。  
- 质量范围 0.7~1.3 倍。

**影响**：抗模型误差，适应不同零部件质量分布。

---

### 2.4 `randomize_com_positions`
**作用**：随机化刚体质心位置。  
**触发**：`startup`

```python
randomize_com_positions = EventTerm(
    func=mdp.randomize_rigid_body_com,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
    },
)
```

**影响**：提升对装配误差、质心偏移的鲁棒性。

---

### 2.5 `randomize_apply_external_force_torque`
**作用**：在 reset 时随机施加外力/外矩扰动。  
**触发**：`reset`

```python
randomize_apply_external_force_torque = EventTerm(
    func=mdp.apply_external_force_torque,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=""),
        "force_range": (-10.0, 10.0),
        "torque_range": (-10.0, 10.0),
    },
)
```

**影响**：改善抗外部扰动能力（起步抗冲击）。

---

### 2.6 `randomize_reset_joints`
**作用**：重置关节位置/速度时添加随机扰动（或缩放）。  
**触发**：`reset`

```python
randomize_reset_joints = EventTerm(
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
        "position_range": (1.0, 1.0),
        "velocity_range": (0.0, 0.0),
    },
)
```

当前参数为固定值（无随机），可以扩展为范围以增加初始化多样性。

---

### 2.7 `randomize_actuator_gains`
**作用**：随机化关节驱动器刚度/阻尼（控制器增益）。  
**触发**：`reset`

```python
randomize_actuator_gains = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (0.5, 2.0),
        "damping_distribution_params": (0.5, 2.0),
        "operation": "scale",
        "distribution": "uniform",
    },
)
```

**影响**：提高对电机/驱动参数误差的容忍度。

---

### 2.8 `randomize_reset_base`
**作用**：在 reset 时随机化 base 位置/速度。  
**触发**：`reset`

```python
randomize_reset_base = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
        },
    },
)
```

**影响**：避免过拟合某个初始状态，提高策略稳健性。

---

### 2.9 `randomize_push_robot`
**作用**：训练中定时推挤机器人（速度扰动）。  
**触发**：`interval`

```python
randomize_push_robot = EventTerm(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(10.0, 15.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
)
```

**影响**：提升对外力干扰的鲁棒性，增强恢复能力。

---

## 3. 实战建议

- 若机器人容易打滑，可适当 **调高摩擦系数下限** 或减小 `push` 幅度。  
- 若策略对真实机器人不稳定，可 **增大质量/质心随机范围**。  
- 若收敛困难，可先降低随机化强度再逐步提升。

---

## 4. 小结
`EventCfg` 是鲁棒性训练的核心模块，通过在不同阶段施加随机化与扰动，使策略具有更强的泛化与抗干扰能力。