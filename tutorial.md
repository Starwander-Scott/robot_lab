# 🐾 零基础小白的 Sim2Sim 强化学习迁移指南：从 IsaacLab 到 MuJoCo

这是一份写给初学者的详尽总结。在这份教程中，我们将复盘把一只在 **IsaacLab**（基于GPU大规模并行的强化学习环境）里训练好的 Unitree Go2 机器狗，成功转移到 **MuJoCo**（高精度高物理保真度的经典模拟器）中运行的全过程。

这个过程在机器人领域被称为 **Sim2Sim (Simulation-to-Simulation) Transfer**。它是前往真实物理世界（Sim2Real）之前的最重要的一步试金石。

---

## 🧐 为什么要做 Sim2Sim？

在 IsaacLab 中训练很快（几千个机器人同时在GPU里跑），但它为了运行速度，牺牲了一些物理底层细节，而且它的驱动器是高度抽象的。
而 MuJoCo 以运算精准、严格遵守物理法则著称。如果你的策略（Policy）在 MuJoCo 这样严苛的环境里也能走得很稳，那么把它部署到真实机器狗上时，一次性成功的概率就会大幅提升。

要想完成迁移，我们需要做以下“移花接木”的工作。

---

## 🛠️ 第一步：提取机器人的“大脑” (Export Policy)

训练结束后，RL框架（如 RSL-RL）往往会保存一个非常臃肿的模型（包含 Actor、Critic 和各类优化器状态）。但在测试和部署时，我们**只需要控制行动的 Actor 网络**。

**我们做了什么：**
我们编写了 `scripts/mujoco/export_go2_jit.py`。其核心是将模型拆解，并用纯 PyTorch 定义一个只包含多层感知机（MLP）的简单网络，并将它打包为脱离 Python 的**TorchScript**，方便跨平台部署：

```python
class ActorMLP(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim, activation):
        super().__init__()
        # ... 构建网络层 (例如 45 -> 512 -> 256 -> 128 -> 12) ...
        self.actor = nn.Sequential(*layers)

# 1. 加载包含全部训练参数的 pt 文件
checkpoint = torch.load("model_xxxx.pt", map_location="cpu")
state = checkpoint["model_state_dict"]

# 2. 仅过滤抽出 'actor.' 开头的权重，丢弃 Critic 等没用的结构
actor_state = {k: v for k, v in state.items() if k.startswith("actor.")}
actor.load_state_dict(actor_state, strict=True)

# 3. 导出成 C++ 等底层支持的 TorchScript 格式
scripted_actor = torch.jit.script(actor)
scripted_actor.save("policy_mujoco.pt")
```

---

## 🌍 第二步：搭建物理世界舞台 (MuJoCo Scene)

MuJoCo 需要通过 XML 文件（MJCF）来了解物理世界。一开始，我们的狗是从半空中“掉进无尽深渊”的，因为导入的 `go2.xml` 里只包含机器狗的本体，并没有定义天地环境。

**我们做了什么：**
设计了 `assets/go2/scene.xml` 嵌套机器狗模型，将其作为主舞台。不仅加入了天空与光照，最重要的是**铺设了实心地板并将楼梯和障碍物删除**，以验证基础走步的稳定性：

```xml
<mujoco model="go2 scene">
  <!-- 引入狗的代码 -->
  <include file="go2.xml"/>

  <!-- 天空与光线 -->
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" .../>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <!-- 这行代码是重中之重，提供带摩擦力的坚固地板 -->
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
```

---

## 👁️ 第三步：建立统一的“感官” (Observation Alignment)

这是 Sim2Sim 最容易失败的一步。神经网络就像一个瞎子，你必须**以它训练时完全一致的格式（45维数组）和顺序**喂给它传感器数据。另外，IsaacLab 中往往包含对输入的自动缩放（Scales），在 MuJoCo 测这套数据时需要手动缩放回去。

**我们做了什么：**
在 `scripts/mujoco/verify_go2_policy.py` 中的 `_build_obs` 函数，我们将 MuJoCo 状态映射成与训练时一致的数组：

```python
def _build_obs_...:
    # 1. 底盘旋转矩阵：将重力和速度投影到机器狗的局部坐标系
    base_xmat = data.xmat[body_id].copy() 
    projected_gravity = _projected_gravity(base_xmat)
    
    # 2. 角速度、关节位置、速度等，必须乘上训练时配置文件设定的缩放比例 (Scales)
    base_ang_vel = _base_ang_vel_body(data, body_id, base_xmat) * 0.25
    joint_pos = (data.qpos[qpos_adr].copy() - default_qpos) * 1.0
    joint_vel = data.qvel[dof_adr].copy() * 0.05

    # 3. 按特定顺序拼出45维大素数组（角速度(3), 重力(3), 指令(3), 关节位移(12), 转速(12), 上一次动作(12)）
    obs_parts.extend([
        base_ang_vel, projected_gravity, cmd, joint_pos, joint_vel, last_action
    ])
    return np.concatenate(obs_parts, axis=0).astype(np.float32)
```

---

## ⚡ 第四步：频率转换与“肌肉”控制 (Action & PD Control)

大脑转得慢，但肌肉反应快，必须要让两者协同工作。

**我们做了什么：**
1. **时钟对齐**：神经网络控制频率 `control_dt = 0.02s` (50Hz)，但物理引擎频率 `sim_dt = 0.005s` (200Hz)。所以神经网络每思考一次，要让 MuJoCo 连续跑 4 步 (`decimation = 4`)。
2. **PD 阻抗控制循环**：神经网络算出的不是绝对扭矩（Torque），而是期望相对角度（Target Position）。我们在 `_apply_action` 里补全了这套纯粹靠力学公式运作的虚拟肌肉：

```python
def _apply_action(...):
    # 按照 Isaac 配置文件进行不同的降幅处理：hip动作放大系数0.125，腿部其它关节放大系数0.25
    scale = _select_action_scale(joint_names)
    
    # 期望角度 = 默认原姿态 + 网络输出 * 比例
    q_des = default_qpos + action * scale

    # 计算 PD 控制扭矩 (Tau)
    q = data.qpos[qpos_adr].copy()   # 当前关节角度
    qd = data.qvel[dof_adr].copy()   # 当前关节角速度
    kp = 25.0 # 这是配置里设定的弹簧刚度
    kd = 0.5  # 这是配置里设定的阻尼
    
    # 公式：扭矩 = P项(追赶误差) - D项(减缓冲击)
    tau = kp * (q_des - q) - kd * qd
    
    # 根据马达最大扭矩夹断数值（Clipping），防止爆炸
    tau = np.clip(tau, ctrl_min, ctrl_max)
    data.ctrl[: len(joint_names)] = tau
```

---

## 💥 第五步：致命的坠落崩溃与解决 (Initialization Crash)

**遇到的最大麻烦：机器人落地疯狂鬼畜、乱振乱摔。**

**为什么？**
在 IsaacLab 训练时，狗天生就是以半蹲姿态出生的（这叫 Default Joint Positions：Hip `0.0`, Thigh `0.8`, Calf `-1.5`）。
但在 MuJoCo 刚启动的第 0 帧世界里，所有关节会被**强行初始化为直线 `0.0` 度**。
因为 Kp(25.0) 是极高的倍率乘数，一开始偏差（从0.0直接变为指令中的0.8）就导致瞬间生成了几百牛米的扭矩。巨大的反作用力瞬间撕裂了系统里的物理向量，`data.qvel` 原地飞天，狗进入死循环乱飞。

**我们在主循环外做了什么：**
强行干预 MuJoCo 底层，给每一个自由度赋予了“初始化待命姿势”：

```python
# 找到训练时候狗半蹲的精确初始角度
default_qpos = _get_default_qpos(model, qpos_adr, GO2_JOINT_NAMES)

# 关键修复代码：Initialize MuJoCo qpos to the default qpos to avoid huge PD impulses
for i, adr in enumerate(qpos_adr):
    data.qpos[adr] = default_qpos[i]

policy = _load_policy(policy_path, args.device)
```
有了这段代码保护，小狗即便在空中下坠时，它的腿也犹如上了发条一般紧紧锁定成正确的倒 V 形，平稳落地。

---

## 🎮 第六步：命令行参数纠错与平滑退出 (UX & Arguments)

引擎能够顺畅推演了，我们需要让它的行为可控又美观。

**我们做了什么：**
1. **指令修正**：之前的命令行输入 `--cmd-vy 0.4` 后狗会“斜向前45度走”，因为 `argparse` 中 `--cmd-vx` 默认值是 `0.5` 并在后台叠加了。我们将默认的前进速度归零。
```python
# 修改前 
parser.add_argument("--cmd-vx", type=float, default=0.5, help="Commanded forward velocity")
# 修改后 
parser.add_argument("--cmd-vx", type=float, default=0.0, help="Commanded forward velocity")
```

2. **解决退出崩溃**：当我们手动关掉 MuJoCo 渲染页面（Viewer）时，终端会爆出 `段错误 (核心已转储)` 的系统问题。这是因为 C++ 底层渲染资源被强行杀掉没有给显卡缓神的余地。
```python
    try:
        step = 0
        while step < total_steps:
            # 引入：检测界面是否被咱们人工 [X] 关掉了，防止在界面消失后代码继续瞎算。
            if viewer is not None and not viewer.is_running():
                break
            ...
    finally:
        # 在退出循环时，给OpenGL上下文进程 0.2秒 的垃圾回收缓冲期
        if viewer is not None:
            if viewer.is_running():
                viewer.close()
            time.sleep(0.2)  
```

---

## 🎉 总结

完成一次出色的 Sim2Sim，就像给不同位面的大跨度物理引擎移植同一套灵魂。不仅要想尽办法**格式化感官数据 (Obs Configuration)**，同步**时间的流动速度 (Control DT)**，最容易忽视但也最致命的一环是**匹配出生点 (Initialization Qpos)** 避免力学爆炸崩溃。

只要掌握了以上 6 步底层原理与代码逻辑，未来你就可以尝试部署拥有几百维数据（包括激光雷达深度测量 Height Scan 和摄像头）的前沿强化学习模型了！