import mujoco

model = mujoco.MjModel.from_xml_path("assets/go2/scene.xml")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"actuator {i}: {name}")
