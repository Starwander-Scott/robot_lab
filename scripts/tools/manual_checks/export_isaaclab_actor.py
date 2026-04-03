import torch

# ==================== 你的路径 ====================
CHECKPOINT_PATH = "/home/oepr/robot_lab/logs/rsl_rl/magiclab_bot_z1_rough/2026-03-18_14-00-12/model_4999.pt"
OUTPUT_PATH = "policy_jit.pt"
# ===================================================

# 加载模型
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

# ==============================================
# 🔥 直接打印模型里有什么，自动找到完整模型
# ==============================================
print("模型里存储的内容：")
for k in checkpoint.keys():
    print(f" - {k}")

# ==============================================
# ✅ 直接提取训练好的完整模型（ IsaacLab 专用 ）
# ==============================================
if "actor" in checkpoint:
    actor = checkpoint["actor"]
elif "policy" in checkpoint:
    actor = checkpoint["policy"]
elif "model" in checkpoint:
    actor = checkpoint["model"]
else:
    # 直接用保存的完整模型结构
    actor = checkpoint

# 切换到推理模式
actor.eval()

# 导出 TorchScript 模型（MUJOCO 能用！）
scripted = torch.jit.script(actor)
scripted.save(OUTPUT_PATH)

print(f"\n✅ 导出成功！文件：{OUTPUT_PATH}")



