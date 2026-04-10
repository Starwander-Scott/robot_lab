# MuJoCo 场景制作与修改指南

## 文件概览

```
assets/go2/
├── go2.xml                  # GO2 机器狗本体模型（不要修改）
├── assets/                  # OBJ 网格文件（16个零件）
├── scene.xml                # 基础平地场景
├── scene_rough_terrain.xml  # 崎岖地形场景（对标 IsaacLab rough terrain）
├── scene_home.xml           # 家庭室内场景（280 平米）
├── scene_home_300.xml       # 家庭室内场景（308 平米）
├── scene_home_300_nav.xml   # 家庭室内导航/避障场景（308 平米）
├── scene_terrain.xml        # 已有的地形场景（含 hfield）
└── scene_obstacles.xml      # 已有的障碍物场景
```

所有 scene 文件通过 `<include file="go2.xml"/>` 引入机器狗，然后在 `<worldbody>` 中定义环境。

---

## 场景结构解析

每个 scene XML 由 4 个核心部分组成：

```xml
<mujoco model="场景名称">
  <!-- 1. 引入机器狗 -->
  <include file="go2.xml"/>

  <!-- 2. 视觉设置（天空、灯光、相机） -->
  <visual>...</visual>

  <!-- 3. 资源定义（纹理、材质、高度场） -->
  <asset>...</asset>

  <!-- 4. 世界物体（地面、墙壁、家具、障碍物） -->
  <worldbody>...</worldbody>
</mujoco>
```

---

## 如何修改场景

### 1. 添加/移动物体

所有物体都在 `<worldbody>` 中用 `<geom>` 定义。核心属性：

```xml
<geom
  pos="x y z"           <!-- 中心位置（米） -->
  type="box"            <!-- 形状：box / sphere / cylinder / capsule / plane / hfield -->
  size="半x 半y 半z"    <!-- 对box是半尺寸；对cylinder是 "半径 半高" -->
  material="材质名"      <!-- 引用 <asset> 中定义的材质 -->
  euler="rx ry rz"      <!-- 旋转角度（度），或用 quat="w x y z" -->
  friction="滑动 旋转 滚动"  <!-- 摩擦系数 -->
  rgba="r g b a"        <!-- 直接指定颜色（不用material时） -->
/>
```

**示例：在家庭场景中添加一张小桌子**

```xml
<!-- 在 <worldbody> 中添加 -->
<!-- 桌面 -->
<geom pos="3.0 2.5 0.5" type="box" size="0.3 0.3 0.015" material="wood"/>
<!-- 四条腿 -->
<geom pos="2.72 2.22 0.243" type="cylinder" size="0.015 0.243" material="chrome"/>
<geom pos="3.28 2.22 0.243" type="cylinder" size="0.015 0.243" material="chrome"/>
<geom pos="2.72 2.78 0.243" type="cylinder" size="0.015 0.243" material="chrome"/>
<geom pos="3.28 2.78 0.243" type="cylinder" size="0.015 0.243" material="chrome"/>
```

### 2. 修改材质和颜色

在 `<asset>` 中定义材质，然后通过 `material="名称"` 引用：

```xml
<asset>
  <!-- 定义纹理 -->
  <texture type="cube" name="my_tex" builtin="flat"
    rgb1="0.8 0.2 0.2" rgb2="0.7 0.15 0.15" width="128" height="128"/>
  <!-- 定义材质 -->
  <material name="my_mat" texture="my_tex" reflectance="0.3"/>
</asset>

<worldbody>
  <geom pos="1 1 0.1" type="box" size="0.5 0.5 0.1" material="my_mat"/>
</worldbody>
```

**注意**：材质名不能与 `go2.xml` 中已有的重复（已有：`metal`, `black`, `white`, `gray`）。

### 3. 添加地形（Height Field）

高度场可以从 PNG 图片生成不规则地形：

```xml
<asset>
  <!-- 从图片生成：size="半x 半y 最大高度 最小采样间距" -->
  <hfield name="my_terrain" size="2.0 2.0 0.3 0.01" file="my_heightmap.png"/>
</asset>

<worldbody>
  <geom type="hfield" hfield="my_terrain" pos="0 0 0"/>
</worldbody>
```

- PNG 灰度值映射到高度：白=最高，黑=最低
- `size` 的第3个参数控制最大凸起高度
- 也可以用 `nrow`/`ncol` 定义程序化高度场（见 `scene_rough_terrain.xml`）

### 4. 修改物理属性

**摩擦力**：直接在 `<geom>` 上设置

```xml
<!-- 高摩擦地毯 -->
<geom pos="1 1 0.01" type="box" size="1 1 0.01"
  friction="1.5 0.05 0.01" material="carpet"/>

<!-- 光滑瓷砖 -->
<geom pos="3 3 0.005" type="box" size="1 1 0.005"
  friction="0.3 0.01 0.005" material="tile"/>
```

**碰撞分组**：用 `contype` 和 `conaffinity` 控制哪些物体会碰撞

```xml
<!-- 仅作视觉显示，不参与碰撞 -->
<geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
```

### 5. 添加斜坡和楼梯

**斜坡**：用 `euler` 旋转 box

```xml
<!-- 15度上坡 -->
<geom pos="2 0 0.13" type="box" size="0.8 0.5 0.02" euler="15 0 0" material="wood"/>
```

**楼梯**：逐级堆叠 box

```xml
<!-- 每级高5cm，深30cm -->
<geom pos="1.0 0 0.025" type="box" size="0.15 0.5 0.025" material="wood"/>
<geom pos="1.3 0 0.050" type="box" size="0.15 0.5 0.050" material="wood"/>
<geom pos="1.6 0 0.075" type="box" size="0.15 0.5 0.075" material="wood"/>
```

---

## 场景说明

### scene_rough_terrain.xml

对标 IsaacLab 中 `ROUGH_TERRAINS_CFG` 的地形类型，分为 6 个区域：

| 区域 | 方向 | 地形类型 | 难度 |
|------|------|----------|------|
| Zone 1 | +X | 金字塔楼梯（上下）| 中 |
| Zone 2 | +Y | 斜坡（15度上下）| 低 |
| Zone 3 | -Y | 随机碎石地形 | 中 |
| Zone 4 | -X | 梅花桩/间隙 | 高 |
| Zone 5 | +X+Y | 高度场波浪地形 | 中 |
| Zone 6 | +X-Y | 窄走廊+障碍 | 高 |

### scene_home.xml

当前 XML 实际为 20m x 14m 的家庭环境，总面积约 280 平米，包含 6 个功能区：

- 客厅+餐厅
- 厨房
- 工具/储藏间
- 中央走廊
- 主卧
- 卫生间
- 次卧/书房

### scene_home_300.xml

在 `scene_home.xml` 思路上扩展为 22m x 14m，总面积约 308 平米，适合做 Go2 的长距离室内连通性测试：

- 南侧功能带：客厅、厨房、工具间
- 中部 22m 长走廊
- 北侧功能带：主卧、卫生间、书房/次卧
- 保留门槛、小障碍、地毯和不同地面材质，适合验证通过性与稳定性

### scene_home_300_nav.xml

基于 `scene_home_300.xml` 叠加导航与避障元素，更适合验证：

- 长走廊 slalom 穿行
- 房门和转角处的路线选择
- 家具密集区域中的绕障稳定性
- 低矮门槛与小物体对步态策略的干扰

设计原则：

- 不做极端高难地形
- 主要增加绕行、窄通道和家庭杂物干扰
- 保持大部分障碍处于室内真实尺度，适合 sim2sim 验证

### 家庭场景默认出生点

`go2.xml` 中机器人本体默认位于世界坐标原点附近，而家庭场景的外墙也从 `X=0, Y=0` 开始。如果直接使用原点出生，Go2 很容易一加载就贴在西南角墙边。

因此在 `scripts/mujoco/verify_go2_policy.py` 里，以下场景会自动使用更合理的默认出生点：

- `scene_home.xml`
- `scene_home_300.xml`
- `scene_home_300_nav.xml`

当前默认出生点为：

- `x=5.2`
- `y=1.4`
- `z=0.45`
- `yaw=0`

这个位置位于客厅内较开阔区域，适合直接起步验证。

如果需要，也可以在命令行手动覆盖：

- `--spawn-x`
- `--spawn-y`
- `--spawn-z`
- `--spawn-yaw-deg`

同样的出生点逻辑也已经同步到：

- `scripts/mujoco/verify_go2_policy_keyboard.py`

---

## 快速测试

```bash
# 查看场景（需要 mujoco-python）
python3 -c "
import mujoco
import mujoco.viewer
m = mujoco.MjModel.from_xml_path('assets/go2/scene_home_300.xml')
d = mujoco.MjData(m)
mujoco.viewer.launch(m, d)
"
```

```bash
# 查看场景（需要 mujoco-python）
python3 -c "
import mujoco
import mujoco.viewer
m = mujoco.MjModel.from_xml_path('assets/go2/scene_rough_terrain.xml')
d = mujoco.MjData(m)
mujoco.viewer.launch(m, d)
"
```


## 在策略推理中使用

```bash
# 搭配 verify_go2_policy.py 测试策略
python scripts/mujoco/verify_go2_policy.py \
  --mjcf assets/go2/scene_home_300_nav.xml \
  --policy <path_to_jit_model.pt>
```

## 键盘控制运行家庭场景

如果你想在 308 平米家庭导航场景中，用键盘实时控制速度指令，使用：

```bash
cd /home/oepr/robot_lab
/home/oepr/robot_lab/.venv/bin/python scripts/mujoco/verify_go2_policy_keyboard.py \
  --mjcf /home/oepr/robot_lab/assets/go2/scene_home_300_nav.xml \
  --policy /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt \
  --duration 300 \
  --control-mode pd \
  --render
```

如果你想手动指定出生点，使用：

```bash
cd /home/oepr/robot_lab
/home/oepr/robot_lab/.venv/bin/python scripts/mujoco/verify_go2_policy_keyboard.py \
  --mjcf /home/oepr/robot_lab/assets/go2/scene_home_300_nav.xml \
  --policy /home/oepr/robot_lab/logs/rsl_rl/unitree_go2_rough/2026-03-28_11-40-49/policy_mujoco.pt \
  --duration 300 \
  --control-mode pd \
  --spawn-x 5.2 \
  --spawn-y 1.4 \
  --spawn-z 0.45 \
  --spawn-yaw-deg 0 \
  --render
```

键盘控制说明：

- `Up / Down`：前进 / 后退
- `Left / Right`：左移 / 右移
- `Q / E`：左转 / 右转
- `Space`：急停




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
