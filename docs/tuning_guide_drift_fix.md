# Go2 平移走偏调参建议书

**问题描述**：Unitree Go2 机器狗在前后（lin_vel_x）或左右（lin_vel_y）平移运动时，机身发生偏转，无法保持直线行进。

**根因分析**：走偏不是 PPO 算法本身的问题，根本原因在于朝向纠偏信号弱、步态不对称惩罚不足、脚底打滑未受到足够约束。真正有效的修改集中在奖励权重和命令配置，PPO 超参数是辅助手段。

---

## 一、Commands 配置（优先级：高）

**文件**：`rough_env_cfg.py`

### 1.1 `heading_control_stiffness`

```python
# 当前
self.commands.base_velocity = mdp.UniformThresholdVelocityCommandCfg(
    heading_control_stiffness=0.5,
    ...
)

# 建议
self.commands.base_velocity = mdp.UniformThresholdVelocityCommandCfg(
    heading_control_stiffness=1.5,   # 建议范围：1.0 ~ 2.0
    ...
)
```

**原理**：该参数将朝向误差（heading error）转换为 ang_vel_z 速度指令。当前值 `0.5` 意味着即使机身偏转较大，下发的纠偏角速度指令依然很小，策略没有强烈动机去对抗偏转。提高该值后纠偏指令更积极，是修复平移走偏最直接的手段。

> **注意**：不要设置过大（>3.0），否则纠偏过激会导致机身左右摇摆。

---

## 二、奖励权重调整（优先级：高）

**文件**：`rough_env_cfg.py`

### 2.1 `joint_mirror.weight` — 左右对称惩罚

```python
# 当前
self.rewards.joint_mirror.weight = -0.05

# 建议
self.rewards.joint_mirror.weight = -0.2   # 建议范围：-0.15 ~ -0.3
```

**原理**：`joint_mirror` 惩罚对角腿（FR↔RL、FL↔RR）关节位置不对称。横移时走偏的典型原因是左右两侧腿的步态存在系统性偏差，增大该惩罚权重可以直接约束这种不对称性。

---

### 2.2 `track_ang_vel_z_exp.weight` — 朝向跟踪奖励

```python
# 当前
self.rewards.track_ang_vel_z_exp.weight = 1.5

# 建议
self.rewards.track_ang_vel_z_exp.weight = 2.5   # 建议范围：2.0 ~ 3.0
```

**原理**：`heading_command=True` 模式下，ang_vel_z 指令由朝向误差生成，该奖励项直接衡量机器人能否跟上朝向指令。当前权重（1.5）相对于线速度跟踪奖励（3.0）偏低，策略倾向于"完成线速度但放弃朝向"。提高该权重使朝向保持与速度跟踪同等重要。

---

### 2.3 `flat_orientation_l2.weight` — 机身姿态惩罚

```python
# 当前
self.rewards.flat_orientation_l2.weight = 0   # 已被 disable_zero_weight_rewards 置 None

# 建议
self.rewards.flat_orientation_l2.weight = -1.0   # 建议范围：-0.5 ~ -2.0
```

**原理**：该奖励惩罚机身的 roll 和 pitch。横移时若机身发生侧倾，地面摩擦力分布随之改变，进一步加剧走偏。启用该项可在机身开始倾斜时提前给出惩罚信号。

---

### 2.4 `feet_slide.weight` — 脚底打滑惩罚

```python
# 当前
self.rewards.feet_slide.weight = -0.1

# 建议
self.rewards.feet_slide.weight = -0.4   # 建议范围：-0.2 ~ -0.5
```

**原理**：脚底打滑是走偏的直接物理原因之一——某只脚在接触地面期间发生侧滑，产生意外的横向位移。增大该惩罚强迫策略学习更稳固的落脚方式。

---

### 2.5 `feet_air_time_variance.weight` — 腾空时间均匀性

```python
# 当前
self.rewards.feet_air_time_variance.weight = -1.0

# 建议
self.rewards.feet_air_time_variance.weight = -2.0
```

**原理**：四条腿腾空时间不均匀意味着步态不对称，不对称步态在累积效果下会导致路径偏移。

---

## 三、PPO 超参数（优先级：低，辅助调整）

**文件**：`agents/rsl_rl_ppo_cfg.py`

### 3.1 `entropy_coef` — 策略熵系数

```python
# 当前
entropy_coef=0.01

# 建议
entropy_coef=0.005
```

**原理**：熵系数控制策略的随机探索程度。`0.01` 在训练后期会保留一定随机性，表现为动作轻微抖动，在直线行进时可能随机积累成偏转。训练后期（>5000 iterations）适当降低熵系数，使策略更加确定性。

> **注意**：训练前期不要改低，否则策略探索不足容易陷入局部最优。可以在 10000 iterations 之后的 fine-tuning 阶段再做调整。

---

### 3.2 `gamma` — 折扣因子

```python
# 当前
gamma=0.99

# 建议
gamma=0.995
```

**原理**：更大的折扣因子使策略更关注长时域奖励。走直线是一个需要长时间保持的行为，`gamma=0.995` 让策略在更长的时间窗口内感知到偏转带来的路径代价。

---

## 四、调参优先级总结

| 优先级 | 参数 | 文件 | 修改方向 | 预期效果 |
|:---:|---|---|---|---|
| ★★★ | `heading_control_stiffness` | `rough_env_cfg.py` | `0.5 → 1.5` | 最直接的朝向纠偏 |
| ★★★ | `joint_mirror.weight` | `rough_env_cfg.py` | `-0.05 → -0.2` | 约束步态左右不对称 |
| ★★☆ | `track_ang_vel_z_exp.weight` | `rough_env_cfg.py` | `1.5 → 2.5` | 增强朝向跟踪激励 |
| ★★☆ | `feet_slide.weight` | `rough_env_cfg.py` | `-0.1 → -0.4` | 减少脚底打滑 |
| ★☆☆ | `flat_orientation_l2.weight` | `rough_env_cfg.py` | `0 → -1.0` | 防止机身侧倾加剧偏转 |
| ★☆☆ | `feet_air_time_variance.weight` | `rough_env_cfg.py` | `-1.0 → -2.0` | 促进步态均匀 |
| ★☆☆ | `entropy_coef` | `rsl_rl_ppo_cfg.py` | `0.01 → 0.005` | 后期策略更确定（fine-tune 阶段） |
| ★☆☆ | `gamma` | `rsl_rl_ppo_cfg.py` | `0.99 → 0.995` | 关注长时域方向保持 |

---

## 五、建议调参流程

1. **第一步**：只改 `heading_control_stiffness=1.5` 和 `joint_mirror.weight=-0.2`，重新训练，观察 TensorBoard 中 `track_ang_vel_z_exp` 奖励曲线是否提升。

2. **第二步**：若仍有明显走偏，加入 `track_ang_vel_z_exp.weight=2.5` 和 `feet_slide.weight=-0.4`。

3. **第三步**：若机身横移时出现侧倾加重走偏，开启 `flat_orientation_l2.weight=-1.0`。

4. **Fine-tune 阶段**（>10000 iterations）：在已有 checkpoint 上继续训练，此时可降低 `entropy_coef=0.005`。

> **每次只改 1~2 个参数**，避免同时调整多项导致无法判断效果来源。
