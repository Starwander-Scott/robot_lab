# BeyondMimic 工具分析

位于 `scripts/tools/beyondmimic/` 目录的 `beyondmimic` 包含用于转换和查看机器人运动数据的实用工具。其主要目的是从原始的 `.csv` 运动轨迹中准备模仿学习 (imitation learning/mimic) 数据集，将其转换为格式完善的 `.npz` 文件，并使用 Isaac Sim 交互式场景验证转换后的结果。

这些工具专门针对 **Unitree G1 29自由度人形机器人** 接口量身定制，逐帧配置其运动学数值。

## 目录结构
- `scripts/tools/beyondmimic/csv_to_npz.py`
- `scripts/tools/beyondmimic/replay_npz.py`

---

## 组件详情

### 1. `csv_to_npz.py`
**用途:** 从 `.csv` 文件回放原始运动轨迹，并将其处理为适合下游模仿学习任务的 `.npz` 文件。

**主要功能:**
- **插值计算 (Interpolation):** 将运动轨迹点从输入帧率 (默认: 60) 转换为所需的输出帧率 (默认: 50)。位置使用标准线性插值 (LERP)，根节点基座四元数使用球面线性插值 (SLERP)。
- **运动学推导 (Kinematic Derivation):** 通过时间步进梯度 (`torch.gradient`) 内部计算速度。明确计算了根节点机体的线速度和角速度（使用 SO(3) 旋转），以及 29 个局部内部自由度关节的速度。
- **仿真烘焙 (Simulation Baking):** 在 Isaac Sim 中生成一个内部交互式场景以严格写入和配置位置限制。迭代拦截仿真状态而不执行完整的物理模拟步骤（使用 `sim.render()` 而不是 `sim.step()`），严格用于导出状态。
- **输出成果 (Output Artifact):** 保存一个结构化 `.npz` 文件，其中包含以下核心记录数据：`fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w` 和 `body_ang_vel_w`。

**使用方法:**
```bash
python scripts/tools/beyondmimic/csv_to_npz.py -f path_to_input.csv --input_fps 60 --output_fps 50
```

### 2. `replay_npz.py`
**用途:** 一个图形验证工具，作为一个独立的查看器，在 Isaac Sim 内部物理测试输出的 `.npz` 数据集，以确保正确生成了所需格式的动作。

**主要功能:**
- **解耦回放 (Decoupled Playback):** 通过原生覆盖系统状态（`.write_root_state_to_sim` 和 `write_joint_state_to_sim`）来精确回放动作。与转换脚本类似，它依赖于渲染而不是推进真实的物理步进，因此视觉回放能保证与数据集完全一致。
- **相机自动化 (Camera Automation):** 自动控制内部查看器的相机，通过迭代移动相机，使得在此空间中导航的人形机器人自动居中保持在视口中。
- **集成测试 (Integration Check):** 直接作为 `robot_lab.tasks.manager_based.beyondmimic.mdp.MotionLoader` 的载体，允许研究人员确认该数据管道与主任务里的 mimic 训练循环完全兼容。

**使用方法:**
```bash
python scripts/tools/beyondmimic/replay_npz.py -f path_to_motion.npz
```
