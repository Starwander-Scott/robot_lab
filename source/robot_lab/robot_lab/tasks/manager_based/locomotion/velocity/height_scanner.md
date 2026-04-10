# height_scanner 详尽说明（基于本仓库配置）

> 位置：`MySceneCfg.height_scanner`（见 `velocity_env_cfg.py`）

## 1. 用途概览
`height_scanner` 是一个 **RayCaster** 地形扫描传感器，用于在机器人附近采样地形高度/距离信息。
在本仓库中，它主要作为 **`height_scan` 观测** 的数据源，用于提供地形感知：
- **Critic**：在 rough 任务中用于 Asymmetric Actor-Critic 的地形信息输入。
- **Actor**：若开启 `observations.policy.height_scan`，也可直接给策略地形信息。

## 2. 关键配置（当前 rough 配置）
对应代码：

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
```

### 字段含义
- `prim_path`：传感器挂载的 prim（机器人基座）。
- `offset`：传感器相对基座的偏移（这里在 z 方向上方 20m 发射射线）。
- `ray_alignment="yaw"`：射线网格随机器人 **航向** 旋转，对齐机体朝向。
- `pattern_cfg`：射线采样网格参数。
- `mesh_prim_paths`：射线检测的目标网格（此处为 `/World/ground`）。
- `debug_vis`：是否显示射线可视化。

## 3. 采样网格与维度
当前配置：
- `resolution = 0.1`
- `size = [1.6, 1.0]`

网格点数计算：
- $\frac{1.6}{0.1} + 1 = 17$ 个点（x 方向）
- $\frac{1.0}{0.1} + 1 = 11$ 个点（y 方向）
- **总点数 $17\times11=187$**

因此 `height_scan` 的观测维度为 **187**。

> 若更改 `size` 或 `resolution`，维度将随之改变。

## 4. 观测侧的使用位置
在 `ObservationsCfg` 中：

```python
height_scan = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    clip=(-1.0, 1.0),
    scale=1.0,
)
```

- `func=mdp.height_scan`：读取 `height_scanner` 输出并形成观测向量。
- `noise`：加入均匀噪声（policy 组启用、critic 组默认不加）。
- `clip`：裁剪到 $[-1, 1]$。
- `scale`：缩放系数（此处为 1.0）。

## 5. 与 `height_scanner_base` 的区别
`height_scanner_base` 也是 RayCaster，但用于 **奖励/状态估计辅助**（如 `base_height_l2`）：
- 网格更小、分辨率更高：`size=(0.1, 0.1)`, `resolution=0.05`
- 输出维度小（$\frac{0.1}{0.05}+1=3$，$3\times3=9$）
- 关注脚下小范围高度变化

## 6. 常见用途/调整建议
- **想减少观测维度**：增大 `resolution` 或缩小 `size`。
- **想增大感知范围**：增大 `size`。
- **想提升地形细节**：减小 `resolution`（代价是维度和计算量上升）。
- **Actor 是否使用**：在 `observations.policy.height_scan` 设为 `None` 可以禁用地形感知。

## 7. 与训练/策略维度的关系
- Flat 任务通常 **不使用** `height_scan`。
- Rough 任务 critic **使用** `height_scan`，构成不对称观察。
- 若更改扫描网格导致维度变化，需同步调整 **critic 网络输入维度**。

---

如需补充该传感器在 `mdp.height_scan` 中的具体计算逻辑，我可以继续整理对应函数说明。