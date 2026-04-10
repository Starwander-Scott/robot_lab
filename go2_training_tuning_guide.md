



























# Go2 训练调参经验整理

日期：2026-04-04

本文按 **Unitree Go2 在 Isaac Gym / Isaac Lab 中进行强化学习 locomotion 训练** 的场景整理，重点是速度控制、稳定步态、rough terrain 泛化，以及 sim2real 前的常见调参经验。

## 结论摘要

Go2 训练中最常见的瓶颈通常不在 PPO 本身，而在以下几项：

- 奖励设计是否存在“站着不动也能拿高分”的漏洞






















- 课程学习是否正常推进
- 动作尺度和 PD 参数是否过激
- 域随机化和延迟建模是否足够支撑 sim2real

经验上，推荐按以下顺序调：

1. 奖励项与奖励权重
2. `action_scale`、`stiffness`、`damping`
3. command 范围与 curriculum
4. 域随机化与延迟
5. PPO 超参数

## 主要参考资料

- `unitree_rl_gym`
  - https://github.com/unitreerobotics/unitree_rl_gym
- Go2 配置
  - https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/go2/go2_config.py
- `legged_gym` 基础配置
  - https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/base/legged_robot_config.py
- `unitree_rl_lab`
  - https://github.com/unitreerobotics/unitree_rl_lab
- Go2 velocity 环境
  - https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py
- 社区 PR：修复可能导致“不走也高分”的奖励问题
  - https://github.com/unitreerobotics/unitree_rl_lab/pull/80
- 社区 PR：修复 terrain curriculum 推进异常
  - https://github.com/unitreerobotics/unitree_rl_lab/pull/60
- Genesis Go2 locomotion 教程
  - https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/getting_started/locomotion.html
- `legged_gym` 论文
  - https://arxiv.org/abs/2109.11978

## 官方配置里值得重点关注的参数

### 1. 旧版 `unitree_rl_gym` 中 Go2 的关键默认值

从官方公开配置可以看到以下几项：

- `action_scale = 0.25`
- `stiffness = 20`
- `damping = 0.5`
- `sim dt = 0.005`
- `decimation = 4`

对应的 PPO 基础默认值主要是：

- `learning_rate = 1e-3`
- `gamma = 0.99`
- `lam = 0.95`
- `clip_param = 0.2`
- `desired_kl = 0.01`
- `entropy_coef = 0.01`
- `num_learning_epochs = 5`
- `num_mini_batches = 4`

Go2 配置本身对 PPO 的修改不大，这说明官方思路更偏向：

- 先用稳定的 PPO 默认设置
- 把主要精力放在环境、奖励和控制参数上

### 2. 新版 `unitree_rl_lab` 中更强调鲁棒性训练

新版 Go2 velocity 环境里，能看到较明确的鲁棒性设置：

- 并行环境数常见为 `4096`
- `dt = 0.005`
- `decimation = 4`
- episode 长度常见为 `20s`
- 摩擦随机化常见为 `0.3 ~ 1.2`
- 基座附加质量随机化常见为 `-1.0 ~ 3.0 kg`
- 间歇性推搡常见为每 `5 ~ 10s` 触发一次
- 动作空间仍以 `JointPositionAction` 为主，`scale = 0.25`

这反映出新版训练范式的重点：

- 不只是学会走
- 而是尽早在训练中加入扰动和随机化，减少 sim2real gap

## 推荐调参表

下面这份表按“先稳训，再提速，再上真机”来用。

| 类别 | 参数 | 官方默认/常见值 | 推荐起步值 | 出问题时优先怎么改 |
|---|---|---:|---:|---|
| 仿真 | `sim dt` | `0.005` | `0.005` | 一般不动 |
| 控制 | `decimation` | `4` | `4` | 动作发飘才考虑升到 `5` |
| 控制 | 控制频率 | `50 Hz` 左右 | `50 Hz` | 真机不稳时先保持较低频率 |
| 动作 | `action_scale` | `0.25` | `0.20 ~ 0.25` | 抖腿或爆动作时先降到 `0.15 ~ 0.20` |
| PD | `stiffness` | `20` | `20` | 软塌时可升到 `25 ~ 30` |
| PD | `damping` | `0.5` | `0.5 ~ 0.8` | 抽搐、震荡时先加到 `0.8 ~ 1.2` |
| PPO | `learning_rate` | `1e-3` | `1e-3` | 曲线震荡大时降到 `5e-4` |
| PPO | `gamma` | `0.99` | `0.99` | 一般不动 |
| PPO | `lam` | `0.95` | `0.95` | 一般不动 |
| PPO | `clip_range` | `0.2` | `0.2` | 更新过猛时降到 `0.1 ~ 0.15` |
| PPO | `entropy_coef` | `0.01` | `0.005 ~ 0.01` | 乱动时降低，不探索时略升 |
| PPO | `num_learning_epochs` | `5` | `5` | 一般不优先修改 |
| PPO | `num_mini_batches` | `4` | `4` | 一般不优先修改 |
| 训练 | 并行环境数 | `4096` 常见 | `2048 ~ 4096` | 显存不够时降到 `1024 ~ 2048` |
| 训练 | `episode_length` | `20s` 常见 | `15 ~ 20s` | 早期学不会可先缩到 `10 ~ 15s` |
| 指令 | `lin vel x` | 宽范围训练 | `0 ~ 1.0 m/s` 起步 | 学不会先缩小范围 |
| 指令 | `lin vel y` | 宽范围训练 | 早期先关或设很小 | 侧移难学，初期不建议放开 |
| 指令 | `yaw vel` | 宽范围训练 | 小范围起步 | 转向混乱时先缩小 |
| 奖励 | `tracking_lin_vel` | 核心项 | 作为最强奖励之一 | 不走就加它 |
| 奖励 | `tracking_ang_vel` | 核心项 | 中等 | 转向差再加 |
| 奖励 | `orientation penalty` | 常见 | 中等 | 前翻侧翻时增强 |
| 奖励 | `dof_acc penalty` | 常见 | 中等偏小 | 动作太冲时增强 |
| 奖励 | `action_rate penalty` | 常见 | 中等 | 抽搐时增强 |
| 奖励 | `torque penalty` | 常见 | 小到中等 | 过强会让机器人不敢动 |
| 奖励 | `feet_slide penalty` | 新版强调 | 中等 | 真机打滑明显时增强 |
| 奖励 | `foot_clearance` | 谨慎使用 | 小权重或前期关闭 | 容易诱导原地抬腿或不走 |
| 随机化 | friction | `0.3 ~ 1.2` 常见 | 开启 | 泛化差时必须保留 |
| 随机化 | base mass | `-1 ~ 3 kg` 常见 | 开启 | 负载变化场景下很重要 |
| 随机化 | pushes | `5 ~ 10s` 一次 | 第二阶段开启 | 早期先关闭 |
| sim2real | action delay | 社区常用 | `10 ~ 20 ms` | 真机发飘时优先补 |
| sim2real | observation noise | 常用 | 轻度开启 | 真机敏感时适当增大 |

## 推荐起步配置

如果目标是先稳定学会前进，再逐步增强鲁棒性，建议直接从下面这组开始：

- `dt = 0.005`
- `decimation = 4`
- `action_scale = 0.20`
- `stiffness = 20`
- `damping = 0.8`
- `learning_rate = 1e-3`
- `clip_range = 0.2`
- `entropy_coef = 0.005`
- `env_num = 2048 ~ 4096`
- `episode_length = 15 ~ 20s`

command 初始建议：

- `vx: 0 ~ 1.0`
- `vy: 0`
- `yaw rate: 小范围`

reward 初始建议保留：

- `tracking_lin_vel`
- `tracking_ang_vel`
- `orientation penalty`
- `action_rate penalty`
- `dof_acc penalty`
- `feet_slide penalty`

reward 初始建议弱化或关闭：

- `foot_clearance`
- 过强的 `torque penalty`
- 所有可能让“站着不动”也能稳定得分的项

域随机化第一阶段建议开启：

- friction randomization
- base mass randomization

第二阶段再加：

- pushes
- 更强的噪声
- 延迟建模

## 常见问题与调参方向

### 1. 站得住，但不走

优先排查奖励设计，而不是 PPO。

建议依次检查：

- `tracking_lin_vel` 是否太弱
- `torque penalty` 是否过强
- `foot_clearance`、默认姿态、静态站立类奖励是否让“不走”也有较高总分
- command 范围是否过宽

优先改法：

- 增强 `tracking_lin_vel`
- 缩小 command 范围
- 弱化静态姿态类奖励
- 暂时关闭或降低 `foot_clearance`

### 2. 会走，但抖腿、抽搐、动作很冲

优先看控制参数和动作尺度。

优先改法：

- 降低 `action_scale`
- 增大 `damping`
- 增大 `action_rate penalty`
- 增大 `dof_acc penalty`
- 如果仍然震荡，再把 `learning_rate` 降到 `5e-4`

### 3. 训练曲线震荡大，收敛不稳定

先不要急着改很多 PPO 参数。

优先排查：

- reward 是否互相冲突
- command 分布是否过宽
- curriculum 是否推进过快

优先改法：

- 降低 `learning_rate`
- 降低 `entropy_coef`
- 缩小 command 范围
- 检查奖励项之间是否互相拉扯

### 4. 平地能走，rough terrain 学不起来

首先确认 terrain curriculum 真的在推进。

社区 PR 曾指出：

- terrain level 可能出现不合理降级
- 某些情况下会导致 level 长期卡住

优先排查：

- terrain level 是否实际持续上升
- timeout 时是否被错误降级
- command resampling 与期望距离估计是否匹配

优先改法：

- 缩小 command 范围
- 降低早期地形难度
- 增强姿态稳定和防滑相关约束
- 检查 curriculum 逻辑本身是否异常

### 5. 仿真走得很好，真机表现差

这通常不是单一参数导致，而是 sim2real 建模不充分。

优先改法：

- 加入 `10 ~ 20 ms` action delay
- 加入 observation noise
- 扩大 friction 和质量随机化范围
- 强化 `feet_slide` 与接触相关约束

## 一条实用的优先级规则

如果只允许优先调三类东西，建议顺序如下：

1. reward scales
2. `action_scale` 与 PD damping
3. command curriculum

PPO 超参数通常排在后面。

## 经验判断表

- `站着不动`：先查奖励漏洞，尤其是脚抬高、默认姿态、静态稳定类奖励
- `会冲但不稳`：先降 `action_scale`，再增强姿态和动作平滑相关惩罚
- `仿真好真机差`：优先补延迟、噪声、摩擦和质量随机化
- `rough terrain 学不起来`：优先查 curriculum 是否正常推进
- `收敛特别慢`：先查奖励与 curriculum，再考虑 PPO

## 备注

以下判断是基于官方配置和社区实战反馈做出的工程经验总结，不是官方明文保证：

- 出现抖腿、抽搐、爆动作时，通常应先降 `action_scale`，再调 `damping`
- 出现“站着不动也能拿高分”时，最常见根因是奖励项设计偏差，而不是 PPO 失效
- sim2real 差距在 Go2 这类足式机器人上，往往对延迟、接触和随机化极其敏感

## 后续可继续补充的内容

如果后面需要，可以继续在本文上追加：

- `Isaac Gym / legged_gym` 风格的 reward scale 具体推荐值
- `Isaac Lab / unitree_rl_lab` 风格的训练模板
- 按“站不稳 / 不走 / 真机差”分类的系统排障 checklist
