import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("assets/go2/scene.xml")
data = mujoco.MjData(model)
data.qvel[3] = 1.0 # ang vel x
data.qvel[4] = 2.0 # ang vel y
data.qvel[5] = 3.0 # ang vel z
mujoco.mj_step(model, data)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
w_global = data.cvel[body_id][:3]
base_xmat = data.xmat[body_id].reshape(3, 3)
w_local = base_xmat.T @ w_global
print("qvel 3:6", data.qvel[3:6])
print("w_local calculated", w_local)
