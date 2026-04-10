# 速度跟踪奖励调研（track_lin_vel_xy_exp / track_ang_vel_z_exp）

> 位置：`velocity_env_cfg.py` 的 `RewardsCfg` 中

## 1. 两个奖励项的定义与目的

### 1.1 `track_lin_vel_xy_exp`
- **目标**：鼓励机器人在 **水平面** 上的线速度 $(v_x, v_y)$ 接近期望指令。  
- **典型形式**：指数型误差奖励，形式常见为：
  
  $$r_{lin}=\exp\left(-\frac{\|\mathbf{v}_{xy}-\mathbf{v}_{xy}^{cmd}\|^2}{\sigma^2}\right)$$

- **优点**：
  - 小误差时奖励高，误差变大时快速衰减。
  - 比线性误差更稳定，有助于平滑收敛。

### 1.2 `track_ang_vel_z_exp`
- **目标**：鼓励机器人绕 z 轴的角速度 $\omega_z$ 接近期望指令。  
- **典型形式**：

  $$r_{yaw}=\exp\left(-\frac{(\omega_z-\omega_z^{cmd})^2}{\sigma^2}\right)$$

- **作用**：
  - 使机器人能稳定转向（跟随角速度或航向指令）。
  - 避免转向过快或偏航失控。

---

## 2. 当前配置中的参数

在 `RewardsCfg` 中：

```python
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=0.0,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
)

track_ang_vel_z_exp = RewTerm(
    func=mdp.track_ang_vel_z_exp,
    weight=0.0,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
)
```

- `command_name`: 使用 `base_velocity` 指令作为期望目标。
- `std = sqrt(0.25) = 0.5`：对应奖励函数中的 $\sigma$。  
  - $\sigma$ 越大，容许误差越大（奖励下降更慢）。  
  - $\sigma$ 越小，要求更严格（稍有偏差奖励快速下降）。

---

## 3. 对训练的影响

### 3.1 机器人向前走的核心驱动力
- 当 `v_x^{cmd} > 0` 时，最大化 `track_lin_vel_xy_exp` 会迫使机器人提高实际前向速度，从而学会向前走。
- 当 `v_y^{cmd} \neq 0` 时，机器人会学会侧向移动或横移。

### 3.2 转向行为的核心驱动力
- 当 `\omega_z^{cmd} \neq 0` 时，`track_ang_vel_z_exp` 引导机器人学会稳定转弯。

### 3.3 与其他奖励的协同
- 速度跟踪奖励提供 **方向性驱动**；
- 姿态、关节、足端等奖励提供 **稳定性与可实现性约束**。

---

## 4. 训练过程中的“变化”

奖励函数本身结构不变，但奖励数值会随机器人状态变化：

- **跟踪好时**：误差小，指数项接近 1。  
- **跟踪差时**：误差大，指数项趋近 0。  
- **动态变化**：策略在不同指令下调整步态，奖励会随时间上下波动。

---

## 5. 调参建议（常用经验）

1. **机器人反应太慢 / 跟踪差**  
   - 提高 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` 权重。  
   - 或适当增大 `std`（更容易给到奖励）。

2. **机器人抖动或不稳定**  
   - 适当降低速度跟踪权重，提升稳定性惩罚项。

3. **转向不稳定 / 原地打转**  
   - 缩小 `ang_vel_z` 指令范围，或提高姿态/滑移惩罚。

---

## 6. 小结
- `track_lin_vel_xy_exp` 和 `track_ang_vel_z_exp` 是 Go2 运动策略的 **核心驱动奖励**。
- 它们通过“速度跟踪”目标引导机器人实现行走与转向。
- 实际训练中，需要与稳定性、能耗、接触约束等奖励 **协同调参**。