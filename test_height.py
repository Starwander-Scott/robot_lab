with open("scripts/mujoco/verify_go2_policy.py", "r") as f:
    text = f.read()

text = text.replace("step += 1", "step += 1\n            if step % 1000 == 0: print('base height:', data.qpos[2])")

with open("scripts/mujoco/verify_go2_policy_height.py", "w") as f:
    f.write(text)
