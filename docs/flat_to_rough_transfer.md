# Flat → Rough 迁移训练：从平地模型在崎岖地形上继续训练

## 背景与问题

在训练机器狗时，通常先在平地（flat）环境中收敛，再迁移到崎岖地形（rough）继续训练，以加快 rough 训练的收敛速度。但直接 `--resume` 加载 flat checkpoint 会报维度错误，原因如下。

---

## 观测空间分析（以 Unitree GO2 为例）

### Policy（Actor）观测

```
base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
+ joint_pos(12) + joint_vel(12) + actions(12) = 45 dims
```

**GO2 flat 和 rough 的 actor 观测维度完全相同（均为 45 维）。**
两份配置都将 `observations.policy.height_scan = None`，所以 actor 没有地形感知，纯粹靠本体感知（proprioception）运动。

### Critic 观测

| 环境   | 组成                                       | 维度       |
|--------|--------------------------------------------|------------|
| Flat   | 同 actor(45) + base_lin_vel(3)             | **48 维**  |
| Rough  | 同 flat(48) + height_scan(187)             | **235 维** |

Flat 配置在 `__post_init__` 中显式移除了 critic 的 height_scan：
```python
# flat_env_cfg.py
self.observations.critic.height_scan = None
```
Rough 配置保留 height_scan，用于给 critic 提供地形信息（Asymmetric Actor-Critic）。

---

## 维度不匹配的根源

RSL-RL 的 `ActorCritic` 网络结构：

```
actor.0.weight  [512, obs_actor_dim]   ← 第一层，输入维度须匹配 actor obs
critic.0.weight [512, obs_critic_dim]  ← 第一层，输入维度须匹配 critic obs
```

Flat checkpoint 中 `critic.0.weight` 形状为 `[512, 48]`，而 rough 环境需要 `[512, 208]`，直接加载会触发 size mismatch 错误。

---

## 解决方案：Critic 输入层零填充扩展

**核心思路**：height_scan 观测位于 critic obs vector 的末尾，因此只需在 `critic.0.weight` 的列方向末尾追加 160 列全零权重。

- 全零初始化意味着训练初期 critic 对 height_scan 输入的响应为零，相当于"无视"地形信息
- 训练过程中，梯度会逐渐驱动这 160 列权重学习有意义的地形特征
- Actor 权重完整保留，无需任何修改

```
原 critic.0.weight:  [512, 48]
                         ↓  追加 187 列零权重
新 critic.0.weight:  [512, 235]   = [512, 48 | 187 zeros]
                                              ↑
                                         height_scan 部分
                                  GridPatternCfg(resolution=0.1, size=[1.6,1.0])
                                  → 17×11 = 187 射线点
```

---

## 操作步骤

### 第一步：转换 checkpoint

```bash
python scripts/tools/convert_flat_to_rough_go2.py \
  logs/rsl_rl/unitree_go2_flat/<run_folder>/model_<iter>.pt \
  logs/rsl_rl/unitree_go2_rough/from_flat_<iter>/model_rough_init.pt
```

脚本会自动：
- 扩展 `critic.0.weight`：`[512, 48] → [512, 208]`
- 保持 `actor` 所有层不变
- 将 `iter` 计数器重置为 0（让 terrain curriculum 从第 0 级重新开始）

已为 `model_71000.pt` 生成的转换文件位于：
```
logs/rsl_rl/unitree_go2_rough/from_flat_71000/model_rough_init.pt
```

### 第二步：在 Rough 环境中 Resume 训练

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --headless \
  --resume \
  --load_run from_flat_71000 \
  --checkpoint model_rough_init.pt
```

> `--load_run` 对应 `logs/rsl_rl/unitree_go2_rough/` 下的子目录名。

---

## 为什么 iter 重置为 0？

Rough 训练使用 terrain curriculum（难度等级 0→5）。RSL-RL 的 curriculum 机制基于 episode 成功率动态调整地形难度，与 iteration 本身不直接挂钩，但 **iteration=0 能确保从最简单的地形开始**，避免直接跳入高难度地形导致策略崩溃。

---

## 预期训练效果

| 阶段 | 表现 |
|------|------|
| 前期（~1000 iter） | Actor 利用已有的本体感知策略快速在 rough terrain 上稳定站立和行走 |
| 中期（~5000 iter） | Critic 的 height_scan 权重逐渐激活，value estimation 更准确，策略开始适应起伏地形 |
| 后期 | 完整利用地形信息，terrain curriculum 推进至高难度等级 |

相比从零开始训练 rough 环境（通常需要 20,000 iter），迁移训练预计可将收敛时间缩短 30%~50%。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/tools/convert_flat_to_rough_go2.py` | checkpoint 转换脚本 |
| `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/flat_env_cfg.py` | GO2 平地环境配置 |
| `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/rough_env_cfg.py` | GO2 崎岖地形环境配置 |
| `logs/rsl_rl/unitree_go2_rough/from_flat_71000/model_rough_init.pt` | 已转换的初始化 checkpoint |
