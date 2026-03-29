with open("scripts/mujoco/verify_go2_policy.py", "r") as f:
    text = f.read()

text = text.replace(
"""    if body_id < 0:
        raise ValueError(f"Body '{args.base_body}' not found in MJCF")

    _, qpos_adr, dof_adr = _build_joint_indices(model, GO2_JOINT_NAMES)
    default_qpos = _get_default_qpos(model, qpos_adr, GO2_JOINT_NAMES)

    policy = _load_policy(policy_path, args.device)""",
"""    if body_id < 0:
        raise ValueError(f"Body '{args.base_body}' not found in MJCF")

    _, qpos_adr, dof_adr = _build_joint_indices(model, GO2_JOINT_NAMES)
    default_qpos = _get_default_qpos(model, qpos_adr, GO2_JOINT_NAMES)

    # Initialize MuJoCo qpos to the default qpos to avoid huge PD impulses
    for i, adr in enumerate(qpos_adr):
        data.qpos[adr] = default_qpos[i]

    policy = _load_policy(policy_path, args.device)""")

with open("scripts/mujoco/verify_go2_policy.py", "w") as f:
    f.write(text)
