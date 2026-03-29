import numpy as np
# MuJoCo to Isaac
qpos_idx_mujoco_to_isaac = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
qpos_idx_isaac_to_mujoco = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8] # wait, it's symmetric!
print(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])[qpos_idx_mujoco_to_isaac])
