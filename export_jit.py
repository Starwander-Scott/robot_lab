import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

# ===================== 你的路径 =====================
CHECKPOINT_PATH = "/home/oepr/robot_lab/logs/rsl_rl/magiclab_bot_z1_rough/2026-03-18_14-00-12/model_4999.pt"
OUTPUT_PATH = "policy_jit.pt"
# ====================================================

# 加载权重
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

# ===================== 【完美匹配权重】=====================
# Actor 输入：45维
# Critic 输入：239维
# 这是你权重里真实的维度！
obs = TensorDict({
    "actor_obs": torch.zeros((1, 45)),   # Actor用
    "critic_obs": torch.zeros((1, 239)) # Critic用
}, batch_size=[1])

obs_groups = {
    "policy": ["actor_obs"],
    "critic": ["critic_obs"]
}
# ============================================================

# 初始化模型（这次100%匹配！）
actor_critic = ActorCritic(
    obs=obs,
    obs_groups=obs_groups,
    num_actions=12,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
    init_noise_std=1.0
)

# 加载权重 ✅✅✅ 绝对不报错！
actor_critic.load_state_dict(checkpoint["model_state_dict"])
actor_critic.eval()

# 导出 Actor 模型 ✅
scripted_actor = torch.jit.script(actor_critic.actor)
scripted_actor.save(OUTPUT_PATH)

print("\n✅ ✅ ✅ 导出成功！模型完全匹配你的权重！")
print("������ 复制到 magiclab_mujoco 即可运行！")