# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Verify a Go2 RSL-RL policy in MuJoCo with gamepad teleoperation."""

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

try:
    import pygame
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pygame is required for gamepad input: pip install pygame") from exc


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
    control_mode: str = "pd"
    kp: float = 25.0
    kd: float = 0.5


@dataclass(frozen=True)
class GamepadLayout:
    axis_id: dict[str, int]
    button_id: dict[str, int]


GAMEPAD_LAYOUTS = {
    "xbox": GamepadLayout(
        axis_id={"LX": 0, "LY": 1, "RX": 3, "RY": 4, "LT": 2, "RT": 5},
        button_id={"A": 0, "B": 1, "X": 2, "Y": 3, "LB": 4, "RB": 5, "SELECT": 6, "START": 7},
    ),
    "switch": GamepadLayout(
        axis_id={"LX": 0, "LY": 1, "RX": 2, "RY": 3, "LT": 5, "RT": 4},
        button_id={"A": 0, "B": 1, "X": 3, "Y": 4, "LB": 6, "RB": 7, "SELECT": 10, "START": 11},
    ),
}


def _build_joint_indices(model: Any, joint_names: list[str]):
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]  # type: ignore[attr-defined]
    if any(jid < 0 for jid in joint_ids):
        missing = [name for name, jid in zip(joint_names, joint_ids) if jid < 0]
        raise ValueError(f"Missing joints in MJCF: {missing}")

    qpos_adr = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dof_adr = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
    return joint_ids, qpos_adr, dof_adr


def _get_default_qpos(model: Any, qpos_adr: list[int], joint_names: list[str]) -> np.ndarray:
    del model, qpos_adr
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
    rotation = base_xmat.reshape(3, 3)
    return (rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)


def _base_ang_vel_body(data: Any, body_id: int, base_xmat: np.ndarray) -> np.ndarray:
    ang_vel_world = data.cvel[body_id][:3].copy()
    rotation = base_xmat.reshape(3, 3)
    return (rotation.T @ ang_vel_world).astype(np.float32)


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
    base_ang_vel = _base_ang_vel_body(data, body_id, base_xmat) * 0.25
    projected_gravity = _projected_gravity(base_xmat)
    joint_pos = (data.qpos[qpos_adr].copy() - default_qpos) * 1.0
    joint_vel = data.qvel[dof_adr].copy() * 0.05

    obs_parts = []
    if obs_cfg.use_base_lin_vel:
        base_lin_vel_world = data.cvel[body_id][3:6].copy()
        rotation = base_xmat.reshape(3, 3)
        base_lin_vel = (rotation.T @ base_lin_vel_world).astype(np.float32)
        obs_parts.append(base_lin_vel)

    obs_parts.extend([base_ang_vel, projected_gravity, cmd, joint_pos, joint_vel, last_action])

    if obs_cfg.use_height_scan:
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
    scale = _select_action_scale(joint_names)
    q_des = default_qpos + action * scale

    if ctrl_cfg.control_mode == "pos":
        if model.nu < len(joint_names):
            raise ValueError("Not enough actuators for position control")
        data.ctrl[: len(joint_names)] = q_des
        return

    q = data.qpos[qpos_adr].copy()
    qd = data.qvel[dof_adr].copy()
    tau = ctrl_cfg.kp * (q_des - q) - ctrl_cfg.kd * qd

    if model.nu < len(joint_names):
        raise ValueError("Not enough actuators for PD control")

    if model.actuator_ctrlrange is not None and model.actuator_ctrlrange.shape[0] >= len(joint_names):
        ctrl_min = model.actuator_ctrlrange[: len(joint_names), 0]
        ctrl_max = model.actuator_ctrlrange[: len(joint_names), 1]
        tau = np.clip(tau, ctrl_min, ctrl_max)

    data.ctrl[: len(joint_names)] = tau


def _load_policy(policy_path: str, device: str) -> torch.jit.ScriptModule:
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    return policy


def _deadband(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def _clip_trigger(value: float) -> float:
    return max(0.0, value)


def _setup_joystick(device_id: int) -> pygame.joystick.Joystick:
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count <= 0:
        raise RuntimeError("No gamepad detected.")
    if device_id >= joystick_count:
        raise ValueError(f"Requested joystick id {device_id}, but only {joystick_count} device(s) found.")
    joystick = pygame.joystick.Joystick(device_id)
    joystick.init()
    return joystick


def _update_command_from_gamepad(
    joystick: pygame.joystick.Joystick,
    layout: GamepadLayout,
    cmd: np.ndarray,
    max_vx: float,
    max_vy: float,
    max_wz: float,
    deadband: float,
) -> bool:
    pygame.event.pump()

    lx = _deadband(float(joystick.get_axis(layout.axis_id["LX"])), deadband)
    ly = _deadband(float(joystick.get_axis(layout.axis_id["LY"])), deadband)
    rx = _deadband(float(joystick.get_axis(layout.axis_id["RX"])), deadband)

    cmd[0] = -ly * max_vx
    cmd[1] = -lx * max_vy
    cmd[2] = -rx * max_wz

    emergency_stop = joystick.get_button(layout.button_id["A"]) or _clip_trigger(
        float(joystick.get_axis(layout.axis_id["LT"]))
    ) > 0.5
    if emergency_stop:
        cmd[:] = 0.0

    return bool(emergency_stop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Go2 RSL-RL policy in MuJoCo with gamepad teleop")
    parser.add_argument("--mjcf", type=str, required=True, help="Path to Go2 MJCF model (xml)")
    parser.add_argument("--policy", type=str, required=True, help="Path to exported policy.pt")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy")
    parser.add_argument("--cmd-vx", type=float, default=0.0, help="Initial commanded forward velocity")
    parser.add_argument("--cmd-vy", type=float, default=0.0, help="Initial commanded lateral velocity")
    parser.add_argument("--cmd-wz", type=float, default=0.0, help="Initial commanded yaw rate")
    parser.add_argument("--sim-dt", type=float, default=None, help="MuJoCo simulation dt")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Control dt (policy step)")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds")
    parser.add_argument("--control-mode", choices=["pd", "pos"], default="pd")
    parser.add_argument("--kp", type=float, default=25.0, help="PD stiffness (only for pd)")
    parser.add_argument("--kd", type=float, default=0.5, help="PD damping (only for pd)")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer")
    parser.add_argument("--base-body", type=str, default="base_link", help="Base body name in MJCF")
    parser.add_argument("--gamepad-id", type=int, default=0, help="pygame joystick device id")
    parser.add_argument("--gamepad-type", choices=sorted(GAMEPAD_LAYOUTS), default="xbox")
    parser.add_argument("--max-vx", type=float, default=1.0, help="Left stick Y -> forward speed scale")
    parser.add_argument("--max-vy", type=float, default=0.6, help="Left stick X -> lateral speed scale")
    parser.add_argument("--max-wz", type=float, default=1.0, help="Right stick X -> yaw rate scale")
    parser.add_argument("--deadband", type=float, default=0.1, help="Axis deadband")
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

    for i, adr in enumerate(qpos_adr):
        data.qpos[adr] = default_qpos[i]

    policy = _load_policy(policy_path, args.device)
    joystick = _setup_joystick(args.gamepad_id)
    layout = GAMEPAD_LAYOUTS[args.gamepad_type]

    obs_cfg = Go2ObsConfig()
    ctrl_cfg = ControlConfig(control_mode=args.control_mode, kp=args.kp, kd=args.kd)
    last_action = np.zeros(len(GO2_JOINT_NAMES), dtype=np.float32)
    cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)

    sim_steps_per_ctrl = max(1, int(round(args.control_dt / model.opt.timestep)))
    total_steps = int(math.ceil(args.duration / model.opt.timestep))

    viewer = None
    if args.render:
        viewer = mujoco.viewer.launch_passive(model, data)

    print("\n=== GO2 GAMEPAD CONTROL ENABLED ===")
    print(f" gamepad     : {joystick.get_name()} (id={args.gamepad_id}, type={args.gamepad_type})")
    print(" left stick  : vx / vy")
    print(" right stick : wz")
    print(" A or LT     : zero command")
    print("===================================\n")

    try:
        step = 0
        last_print = 0.0
        while step < total_steps:
            if viewer is not None and not viewer.is_running():
                break

            emergency_stop = _update_command_from_gamepad(
                joystick=joystick,
                layout=layout,
                cmd=cmd,
                max_vx=args.max_vx,
                max_vy=args.max_vy,
                max_wz=args.max_wz,
                deadband=args.deadband,
            )

            now = time.time()
            if now - last_print > 0.1:
                stop_suffix = " [STOP]" if emergency_stop else ""
                print(
                    f"\r[Command] vx: {cmd[0]:.2f}, vy: {cmd[1]:.2f}, wz: {cmd[2]:.2f}{stop_suffix}     ",
                    end="",
                )
                last_print = now

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
        if viewer is not None and viewer.is_running():
            viewer.close()
            time.sleep(0.2)
        pygame.quit()
        print()


if __name__ == "__main__":
    main()
