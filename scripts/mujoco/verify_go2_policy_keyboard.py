# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Verify a Go2 RSL-RL policy in MuJoCo (sim2sim)."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("MuJoCo is required: pip install mujoco") from exc


GO2_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# Action scaling from UnitreeGo2RoughEnvCfg
GO2_ACTION_SCALE = {
    ".*_hip_joint": 0.125,                                     
    "^(?!.*_hip_joint).*": 0.25,
}


@dataclass
class Go2ObsConfig:
    use_base_lin_vel: bool = False
    use_height_scan: bool = False


@dataclass
class ControlConfig:
    control_mode: str = "pd"  # "pd" or "pos"  
    kp: float = 25.0
    kd: float = 0.5


def _build_joint_indices(model: Any, joint_names: list[str]):
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]  # type: ignore[attr-defined]
    if any(jid < 0 for jid in joint_ids):
        missing = [name for name, jid in zip(joint_names, joint_ids) if jid < 0]
        raise ValueError(f"Missing joints in MJCF: {missing}")

    qpos_adr = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dof_adr = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
    return joint_ids, qpos_adr, dof_adr


def _get_default_qpos(model: Any, qpos_adr: list[int], joint_names: list[str]) -> np.ndarray:
    # Strictly use Isaac Sim's default joint positions for Go2 to avoid keyframe mismatch
    default_qpos = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        if "hip_" in name:
            default_qpos[i] = 0.0
        elif "thigh_" in name:
            default_qpos[i] = 0.8
        elif "calf_" in name:
            default_qpos[i] = -1.5
    return default_qpos


def _projected_gravity(base_xmat: np.ndarray) -> np.ndarray:
    # base_xmat: (9,) row-major rotation matrix body->world
    R = base_xmat.reshape(3, 3)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (R.T @ g_world).astype(np.float32)


def _base_ang_vel_body(data: Any, body_id: int, base_xmat: np.ndarray) -> np.ndarray:
    # data.cvel is spatial velocity: [ang, lin] in global orientation
    w_global = data.cvel[body_id][:3].copy()
    R = base_xmat.reshape(3, 3)
    w_local = R.T @ w_global
    return w_local.astype(np.float32)


def _select_action_scale(joint_names: list[str]) -> np.ndarray:
    scales = []
    for name in joint_names:
        if "_hip_joint" in name:
            scales.append(GO2_ACTION_SCALE[".*_hip_joint"])
        else:
            scales.append(GO2_ACTION_SCALE["^(?!.*_hip_joint).*"])
    return np.array(scales, dtype=np.float32)


def _build_obs(
    data: Any,
    body_id: int,
    qpos_adr: list[int],
    dof_adr: list[int],
    default_qpos: np.ndarray,
    last_action: np.ndarray,
    cmd: np.ndarray,
    obs_cfg: Go2ObsConfig,
) -> np.ndarray:
    base_xmat = data.xmat[body_id].copy()
    
    # Apply Go2 observation scales from UNITREE_GO2_ROUGH_CFG
    base_ang_vel = _base_ang_vel_body(data, body_id, base_xmat) * 0.25
    projected_gravity = _projected_gravity(base_xmat)

    joint_pos = (data.qpos[qpos_adr].copy() - default_qpos) * 1.0
    joint_vel = data.qvel[dof_adr].copy() * 0.05

    obs_parts = []
    if obs_cfg.use_base_lin_vel:
        # Use cvel instead of xvelp because xvelp is dependent on joints vs free joint
        base_lin_vel_world = data.cvel[body_id][3:6].copy()
        R = base_xmat.reshape(3, 3)
        base_lin_vel = (R.T @ base_lin_vel_world).astype(np.float32)
        obs_parts.append(base_lin_vel)

    obs_parts.extend([
        base_ang_vel,
        projected_gravity,
        cmd,
        joint_pos,
        joint_vel,
        last_action,
    ])

    if obs_cfg.use_height_scan:
        # Dummy height scan matching dimensions (187 points typically)
        obs_parts.append(np.zeros(187, dtype=np.float32))

    return np.concatenate(obs_parts, axis=0).astype(np.float32)

def _apply_action(
    model: Any,
    data: Any,
    joint_names: list[str],
    qpos_adr: list[int],
    dof_adr: list[int],
    default_qpos: np.ndarray,
    action: np.ndarray,
    ctrl_cfg: ControlConfig,
):
    # Map action -> target joint positions
    scale = _select_action_scale(joint_names)
    q_des = default_qpos + action * scale

    if ctrl_cfg.control_mode == "pos":
        # Assume position actuators in MJCF
        if model.nu < len(joint_names):
            raise ValueError("Not enough actuators for position control")
        data.ctrl[: len(joint_names)] = q_des
        return

    # PD torque control
    q = data.qpos[qpos_adr].copy()
    qd = data.qvel[dof_adr].copy()
    tau = ctrl_cfg.kp * (q_des - q) - ctrl_cfg.kd * qd

    if model.nu < len(joint_names):
        raise ValueError("Not enough actuators for PD control")

    # Respect actuator ctrlrange if available
    if model.actuator_ctrlrange is not None and model.actuator_ctrlrange.shape[0] >= len(joint_names):
        ctrl_min = model.actuator_ctrlrange[: len(joint_names), 0]
        ctrl_max = model.actuator_ctrlrange[: len(joint_names), 1]
        tau = np.clip(tau, ctrl_min, ctrl_max)

    data.ctrl[: len(joint_names)] = tau


def _load_policy(policy_path: str, device: str) -> torch.jit.ScriptModule:
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Go2 RSL-RL policy in MuJoCo")
    parser.add_argument("--mjcf", type=str, required=True, help="Path to Go2 MJCF model (xml)")
    parser.add_argument("--policy", type=str, required=True, help="Path to exported policy.pt")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy")
    parser.add_argument("--cmd-vx", type=float, default=0.0, help="Commanded forward velocity")
    parser.add_argument("--cmd-vy", type=float, default=0.0, help="Commanded lateral velocity")
    parser.add_argument("--cmd-wz", type=float, default=0.0, help="Commanded yaw rate")
    parser.add_argument("--sim-dt", type=float, default=None, help="MuJoCo simulation dt")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Control dt (policy step)")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds")
    parser.add_argument("--control-mode", choices=["pd", "pos"], default="pd")
    parser.add_argument("--kp", type=float, default=25.0, help="PD stiffness (only for pd)")
    parser.add_argument("--kd", type=float, default=0.5, help="PD damping (only for pd)")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer")
    parser.add_argument("--base-body", type=str, default="base_link", help="Base body name in MJCF")
    args = parser.parse_args()

    policy_path = os.path.abspath(args.policy)
    mjcf_path = os.path.abspath(args.mjcf)
    mjcf_dir = os.path.dirname(mjcf_path)
    if mjcf_dir:
        os.chdir(mjcf_dir)
    model = mujoco.MjModel.from_xml_path(mjcf_path)  # type: ignore[attr-defined]
    data = mujoco.MjData(model)  # type: ignore[attr-defined]

    if args.sim_dt is not None:
        model.opt.timestep = args.sim_dt

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.base_body)  # type: ignore[attr-defined]
    if body_id < 0:
        raise ValueError(f"Body '{args.base_body}' not found in MJCF")

    _, qpos_adr, dof_adr = _build_joint_indices(model, GO2_JOINT_NAMES)
    default_qpos = _get_default_qpos(model, qpos_adr, GO2_JOINT_NAMES)

    # Initialize MuJoCo qpos to the default qpos to avoid huge PD impulses
    for i, adr in enumerate(qpos_adr):
        data.qpos[adr] = default_qpos[i]

    policy = _load_policy(policy_path, args.device)

    obs_cfg = Go2ObsConfig()
    ctrl_cfg = ControlConfig(control_mode=args.control_mode, kp=args.kp, kd=args.kd)

    last_action = np.zeros(len(GO2_JOINT_NAMES), dtype=np.float32)
    cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)

    # =============== KEYBOARD CALLBACK ===============
    def key_callback(keycode: int):
        # GLFW keycodes: UP=265, DOWN=264, LEFT=263, RIGHT=262, Q=81, E=69, SPACE=32
        step_v = 0.2
        step_w = 0.2
        if keycode == 265:    # UP Arrow (Forward)
            cmd[0] += step_v
        elif keycode == 264:  # DOWN Arrow (Backward)
            cmd[0] -= step_v
        elif keycode == 263:  # LEFT Arrow (Strafe Left)
            cmd[1] += step_v
        elif keycode == 262:  # RIGHT Arrow (Strafe Right)
            cmd[1] -= step_v
        elif keycode == 81:   # Q (Turn left)
            cmd[2] += step_w
        elif keycode == 69:   # E (Turn right)
            cmd[2] -= step_w
        elif keycode == 32:   # Space
            cmd[:] = 0.0

        # Optional clipping to prevent the values getting too huge
        cmd[0] = np.clip(cmd[0], -2.0, 2.0)
        cmd[1] = np.clip(cmd[1], -1.0, 1.0)
        cmd[2] = np.clip(cmd[2], -1.5, 1.5)
        
        print(f"\r[Command] vx: {cmd[0]:.2f}, vy: {cmd[1]:.2f}, wz: {cmd[2]:.2f}     ", end="")

    sim_steps_per_ctrl = max(1, int(round(args.control_dt / model.opt.timestep)))
    total_steps = int(math.ceil(args.duration / (model.opt.timestep)))

    viewer = None
    if args.render:
        print("\n=== GO2 KEYBOARD CONTROL ENABLED ===")
        print(" [UP / DOWN]    : Forward / Backward")
        print(" [LEFT / RIGHT] : Left / Right (Strafe)")
        print(" [Q / E]        : Turn Left / Turn Right")
        print(" [SPACE]        : Emergency Stop")
        print("====================================\n")
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    try:
        step = 0
        while step < total_steps:
            if viewer is not None and not viewer.is_running():
                break

            if step % sim_steps_per_ctrl == 0:
                obs = _build_obs(
                    data,
                    body_id,
                    qpos_adr,
                    dof_adr,
                    default_qpos,
                    last_action,
                    cmd,
                    obs_cfg,
                )
                obs_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                with torch.inference_mode():
                    action = policy(obs_t).cpu().numpy().squeeze(0)
                last_action = action.astype(np.float32)
                _apply_action(
                    model,
                    data,
                    GO2_JOINT_NAMES,
                    qpos_adr,
                    dof_adr,
                    default_qpos,
                    last_action,
                    ctrl_cfg,
                )

            mujoco.mj_step(model, data)  # type: ignore[attr-defined]

            if viewer is not None:
                viewer.sync()

            step += 1
    finally:
        if viewer is not None:
            if viewer.is_running():
                viewer.close()
            time.sleep(0.2)  # Give OpenGL context time to cleanly shutdown


if __name__ == "__main__":
    main()
