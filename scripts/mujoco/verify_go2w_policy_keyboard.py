# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Verify a Go2W RSL-RL policy in MuJoCo with keyboard teleoperation."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("MuJoCo is required: pip install mujoco") from exc


GO2W_LEG_JOINT_NAMES = [
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

GO2W_WHEEL_JOINT_NAMES = [
    "FR_foot_joint",
    "FL_foot_joint",
    "RR_foot_joint",
    "RL_foot_joint",
]

GO2W_WHEEL_JOINT_ALIASES = [
    GO2W_WHEEL_JOINT_NAMES,
    ["FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint"],
]

GO2W_JOINT_NAMES = GO2W_LEG_JOINT_NAMES + GO2W_WHEEL_JOINT_NAMES

GO2W_LEG_ACTION_SCALE = {
    ".*_hip_joint": 0.125,
    "^(?!.*_hip_joint).*": 0.25,
}
GO2W_WHEEL_ACTION_SCALE = 5.0


@dataclass
class ControlConfig:
    control_mode: str = "pd"
    leg_kp: float = 25.0
    leg_kd: float = 0.5
    wheel_kd: float = 0.5


@dataclass
class SpawnConfig:
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int], action_dim: int, activation: str) -> None:
        super().__init__()
        activations = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        dims = [obs_dim] + hidden_dims + [action_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activations[activation])
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def _build_joint_indices(model: Any, joint_names: list[str], mjcf_path: str):
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]  # type: ignore[attr-defined]
    if any(jid < 0 for jid in joint_ids):
        missing = [name for name, jid in zip(joint_names, joint_ids) if jid < 0]
        raise ValueError(
            "Missing joints in MJCF: "
            f"{missing}. The Go2W policy requires a wheeled MJCF with the four '*_foot_joint' wheel joints. "
            f"Current file: {mjcf_path}"
        )

    qpos_adr = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dof_adr = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
    return joint_ids, qpos_adr, dof_adr


def _resolve_wheel_joint_names(model: Any, mjcf_path: str) -> list[str]:
    for joint_names in GO2W_WHEEL_JOINT_ALIASES:
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]  # type: ignore[attr-defined]
        if all(jid >= 0 for jid in joint_ids):
            return joint_names

    expected = [" / ".join(names[i] for names in GO2W_WHEEL_JOINT_ALIASES) for i in range(len(GO2W_WHEEL_JOINT_NAMES))]
    raise ValueError(
        "Missing wheel joints in MJCF: "
        f"{expected}. The Go2W policy requires four wheel joints. Current file: {mjcf_path}"
    )


def _get_default_qpos(joint_names: list[str]) -> np.ndarray:
    default_qpos = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        if "hip_" in name:
            default_qpos[i] = 0.0
        elif "thigh_" in name:
            default_qpos[i] = 0.8
        elif "calf_" in name:
            default_qpos[i] = -1.5
        elif "foot_" in name or "wheel_" in name:
            default_qpos[i] = 0.0
    return default_qpos


def _projected_gravity(base_xmat: np.ndarray) -> np.ndarray:
    rotation = base_xmat.reshape(3, 3)
    return (rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)


def _base_ang_vel_body(data: Any, body_id: int, base_xmat: np.ndarray) -> np.ndarray:
    ang_vel_world = data.cvel[body_id][:3].copy()
    rotation = base_xmat.reshape(3, 3)
    return (rotation.T @ ang_vel_world).astype(np.float32)


def _leg_action_scale(joint_names: list[str]) -> np.ndarray:
    scales = []
    for name in joint_names:
        if "_hip_joint" in name:
            scales.append(GO2W_LEG_ACTION_SCALE[".*_hip_joint"])
        else:
            scales.append(GO2W_LEG_ACTION_SCALE["^(?!.*_hip_joint).*"])
    return np.array(scales, dtype=np.float32)


def _build_obs(
    data: Any,
    body_id: int,
    qpos_adr: list[int],
    dof_adr: list[int],
    default_qpos: np.ndarray,
    last_action: np.ndarray,
    cmd: np.ndarray,
) -> np.ndarray:
    base_xmat = data.xmat[body_id].copy()
    base_ang_vel = _base_ang_vel_body(data, body_id, base_xmat) * 0.25
    projected_gravity = _projected_gravity(base_xmat)

    joint_pos = data.qpos[qpos_adr].copy() - default_qpos
    joint_vel = data.qvel[dof_adr].copy() * 0.05

    # Match joint_pos_rel_without_wheel used in UnitreeGo2WFlatEnvCfg.
    joint_pos[-len(GO2W_WHEEL_JOINT_NAMES) :] = 0.0

    obs = np.concatenate(
        [
            base_ang_vel,
            projected_gravity,
            cmd,
            joint_pos.astype(np.float32),
            joint_vel.astype(np.float32),
            last_action.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return obs


def _apply_action(
    model: Any,
    data: Any,
    leg_qpos_adr: list[int],
    leg_dof_adr: list[int],
    wheel_dof_adr: list[int],
    default_leg_qpos: np.ndarray,
    action: np.ndarray,
    ctrl_cfg: ControlConfig,
):
    leg_action = action[: len(GO2W_LEG_JOINT_NAMES)]
    wheel_action = action[len(GO2W_LEG_JOINT_NAMES) :]

    leg_scale = _leg_action_scale(GO2W_LEG_JOINT_NAMES)
    leg_q_des = default_leg_qpos + leg_action * leg_scale
    wheel_qd_des = wheel_action * GO2W_WHEEL_ACTION_SCALE

    if model.nu < len(GO2W_JOINT_NAMES):
        raise ValueError(f"Not enough actuators for Go2W control: expected >= {len(GO2W_JOINT_NAMES)}, got {model.nu}")

    ctrl = np.zeros(len(GO2W_JOINT_NAMES), dtype=np.float32)
    if ctrl_cfg.control_mode == "pos":
        ctrl[: len(GO2W_LEG_JOINT_NAMES)] = leg_q_des
        ctrl[len(GO2W_LEG_JOINT_NAMES) :] = wheel_qd_des
    else:
        leg_q = data.qpos[leg_qpos_adr].copy()
        leg_qd = data.qvel[leg_dof_adr].copy()
        wheel_qd = data.qvel[wheel_dof_adr].copy()

        leg_tau = ctrl_cfg.leg_kp * (leg_q_des - leg_q) - ctrl_cfg.leg_kd * leg_qd
        wheel_tau = ctrl_cfg.wheel_kd * (wheel_qd_des - wheel_qd)

        ctrl[: len(GO2W_LEG_JOINT_NAMES)] = leg_tau
        ctrl[len(GO2W_LEG_JOINT_NAMES) :] = wheel_tau

    if model.actuator_ctrlrange is not None and model.actuator_ctrlrange.shape[0] >= len(GO2W_JOINT_NAMES):
        ctrl_min = model.actuator_ctrlrange[: len(GO2W_JOINT_NAMES), 0]
        ctrl_max = model.actuator_ctrlrange[: len(GO2W_JOINT_NAMES), 1]
        ctrl = np.clip(ctrl, ctrl_min, ctrl_max)

    data.ctrl[: len(GO2W_JOINT_NAMES)] = ctrl


def _load_policy(policy_path: str, device: str, activation: str) -> nn.Module:
    try:
        policy = torch.jit.load(policy_path, map_location=device)
        policy.eval()
        return policy
    except RuntimeError:
        pass

    checkpoint = torch.load(policy_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported policy format: {policy_path}")

    state = checkpoint["model_state_dict"]
    actor_keys = [key for key in state if key.startswith("actor.")]
    if not actor_keys:
        raise ValueError(f"No actor weights found in checkpoint: {policy_path}")

    actor_state = {key: value for key, value in state.items() if key.startswith("actor.")}
    first_weight = actor_state["actor.0.weight"]
    last_weight_key = sorted((key for key in actor_state if key.endswith(".weight")), key=lambda item: int(item.split(".")[1]))[-1]
    last_weight = actor_state[last_weight_key]
    hidden_dims = []
    weight_keys = sorted((key for key in actor_state if key.endswith(".weight")), key=lambda item: int(item.split(".")[1]))
    for key in weight_keys[:-1]:
        hidden_dims.append(int(actor_state[key].shape[0]))

    actor = ActorMLP(
        obs_dim=int(first_weight.shape[1]),
        hidden_dims=hidden_dims,
        action_dim=int(last_weight.shape[0]),
        activation=activation,
    )
    actor.load_state_dict(actor_state, strict=True)
    actor.to(device)
    actor.eval()
    return actor


def _infer_spawn_config(mjcf_path: str) -> SpawnConfig:
    scene_name = os.path.basename(mjcf_path)
    if scene_name in {"scene_home.xml", "scene_home_300.xml", "scene_home_300_nav.xml"}:
        return SpawnConfig(x=6.2, y=1.0, z=0.45, yaw_deg=0.0)
    return SpawnConfig(x=0.0, y=0.0, z=0.45, yaw_deg=0.0)


def _apply_spawn_pose(data: Any, spawn_cfg: SpawnConfig) -> None:
    yaw = math.radians(spawn_cfg.yaw_deg)
    data.qpos[0] = spawn_cfg.x
    data.qpos[1] = spawn_cfg.y
    data.qpos[2] = spawn_cfg.z
    data.qpos[3] = math.cos(yaw / 2.0)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = math.sin(yaw / 2.0)
    data.qvel[:6] = 0.0


def _respawn_robot(
    data: Any,
    home_qpos: np.ndarray,
    home_qvel: np.ndarray,
    last_action: np.ndarray,
    cmd: np.ndarray,
) -> None:
    data.qpos[:] = home_qpos
    data.qvel[:] = home_qvel
    data.ctrl[:] = 0.0
    if getattr(data, "act", None) is not None and data.act.size > 0:
        data.act[:] = 0.0
    if getattr(data, "qacc_warmstart", None) is not None and data.qacc_warmstart.size > 0:
        data.qacc_warmstart[:] = 0.0
    last_action[:] = 0.0
    cmd[:] = 0.0


def _lift_robot(
    model: Any,
    data: Any,
    lift_height: float,
    last_action: np.ndarray,
    cmd: np.ndarray,
) -> None:
    data.qpos[2] += lift_height
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    if getattr(data, "act", None) is not None and data.act.size > 0:
        data.act[:] = 0.0
    if getattr(data, "qacc_warmstart", None) is not None and data.qacc_warmstart.size > 0:
        data.qacc_warmstart[:] = 0.0
    last_action[:] = 0.0
    cmd[:] = 0.0
    mujoco.mj_forward(model, data)  # type: ignore[attr-defined]


def _resolve_base_body(model: Any, requested_name: str | None) -> tuple[int, str]:
    candidate_names = [requested_name] if requested_name else []
    candidate_names.extend(["base", "base_link"])

    for name in candidate_names:
        if name is None:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # type: ignore[attr-defined]
        if body_id >= 0:
            return body_id, name

    raise ValueError(f"Could not find base body. Tried: {candidate_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Go2W RSL-RL policy in MuJoCo with keyboard teleop")
    parser.add_argument("--mjcf", type=str, required=True, help="Path to Go2W MJCF model (xml)")
    parser.add_argument("--policy", type=str, required=True, help="Path to exported TorchScript or RSL-RL model_XXXX.pt")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy")
    parser.add_argument("--activation", type=str, default="elu", help="Activation for raw RSL-RL checkpoints")
    parser.add_argument("--cmd-vx", type=float, default=0.0, help="Initial commanded forward velocity")
    parser.add_argument("--cmd-vy", type=float, default=0.0, help="Initial commanded lateral velocity")
    parser.add_argument("--cmd-wz", type=float, default=0.0, help="Initial commanded yaw rate")
    parser.add_argument("--sim-dt", type=float, default=None, help="MuJoCo simulation dt")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Control dt (policy step)")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds")
    parser.add_argument("--control-mode", choices=["pd", "pos"], default="pd")
    parser.add_argument("--leg-kp", type=float, default=25.0, help="Leg PD stiffness")
    parser.add_argument("--leg-kd", type=float, default=0.5, help="Leg PD damping")
    parser.add_argument("--wheel-kd", type=float, default=0.5, help="Wheel velocity damping/gain")
    parser.add_argument("--spawn-x", type=float, default=None, help="Override base spawn x in world frame")
    parser.add_argument("--spawn-y", type=float, default=None, help="Override base spawn y in world frame")
    parser.add_argument("--spawn-z", type=float, default=None, help="Override base spawn z in world frame")
    parser.add_argument("--spawn-yaw-deg", type=float, default=None, help="Override base spawn yaw in degrees")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer")
    parser.add_argument("--base-body", type=str, default=None, help="Base body name in MJCF, auto-detect if omitted")
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

    body_id, body_name = _resolve_base_body(model, args.base_body)
    wheel_joint_names = _resolve_wheel_joint_names(model, mjcf_path)
    joint_names = GO2W_LEG_JOINT_NAMES + wheel_joint_names
    _, qpos_adr, dof_adr = _build_joint_indices(model, joint_names, mjcf_path)
    leg_qpos_adr = qpos_adr[: len(GO2W_LEG_JOINT_NAMES)]
    leg_dof_adr = dof_adr[: len(GO2W_LEG_JOINT_NAMES)]
    wheel_dof_adr = dof_adr[len(GO2W_LEG_JOINT_NAMES) :]

    default_qpos = _get_default_qpos(joint_names)
    default_leg_qpos = default_qpos[: len(GO2W_LEG_JOINT_NAMES)]

    spawn_cfg = _infer_spawn_config(mjcf_path)
    if args.spawn_x is not None:
        spawn_cfg.x = args.spawn_x
    if args.spawn_y is not None:
        spawn_cfg.y = args.spawn_y
    if args.spawn_z is not None:
        spawn_cfg.z = args.spawn_z
    if args.spawn_yaw_deg is not None:
        spawn_cfg.yaw_deg = args.spawn_yaw_deg

    _apply_spawn_pose(data, spawn_cfg)
    for i, adr in enumerate(qpos_adr):
        data.qpos[adr] = default_qpos[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # type: ignore[attr-defined]

    home_qpos = data.qpos.copy()
    home_qvel = data.qvel.copy()

    policy = _load_policy(policy_path, args.device, args.activation)
    ctrl_cfg = ControlConfig(
        control_mode=args.control_mode,
        leg_kp=args.leg_kp,
        leg_kd=args.leg_kd,
        wheel_kd=args.wheel_kd,
    )

    last_action = np.zeros(len(joint_names), dtype=np.float32)
    cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)
    respawn_requested = False
    lift_requested = False
    last_respawn_time = -1.0
    last_lift_time = -1.0

    def key_callback(keycode: int):
        nonlocal respawn_requested, lift_requested, last_respawn_time, last_lift_time
        step_v = 0.2
        step_w = 0.2
        if keycode == 265:
            cmd[0] += step_v
        elif keycode == 264:
            cmd[0] -= step_v
        elif keycode == 263:
            cmd[1] += step_v
        elif keycode == 262:
            cmd[1] -= step_v
        elif keycode == 81:
            cmd[2] += step_w
        elif keycode == 69:
            cmd[2] -= step_w
        elif keycode == 32:
            cmd[:] = 0.0
        elif keycode == 48:
            now = time.monotonic()
            if now - last_respawn_time >= 0.5:
                last_respawn_time = now
                respawn_requested = True
                return
        elif keycode == 57:
            now = time.monotonic()
            if now - last_lift_time >= 0.3:
                last_lift_time = now
                lift_requested = True
                return

        cmd[0] = np.clip(cmd[0], -2.0, 2.0)
        cmd[1] = np.clip(cmd[1], -1.0, 1.0)
        cmd[2] = np.clip(cmd[2], -1.5, 1.5)
        print(f"\r[Command] vx: {cmd[0]:.2f}, vy: {cmd[1]:.2f}, wz: {cmd[2]:.2f}     ", end="")

    sim_steps_per_ctrl = max(1, int(round(args.control_dt / model.opt.timestep)))
    total_steps = int(math.ceil(args.duration / model.opt.timestep))

    viewer = None
    if args.render:
        print("\n=== GO2W KEYBOARD CONTROL ENABLED ===")
        print(f" Base body: {body_name}")
        print(" [UP / DOWN]    : Forward / Backward")
        print(" [LEFT / RIGHT] : Left / Right (Strafe)")
        print(" [Q / E]        : Turn Left / Turn Right")
        print(" [SPACE]        : Emergency Stop")
        print(" [0]            : Respawn at spawn point")
        print(" [9]            : Lift robot upward by 0.1m")
        print("=====================================\n")
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        viewer.opt.geomgroup[:] = 1

    try:
        step = 0
        while step < total_steps:
            if viewer is not None and not viewer.is_running():
                break

            if respawn_requested:
                _respawn_robot(data, home_qpos, home_qvel, last_action, cmd)
                mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
                respawn_requested = False
                if viewer is not None:
                    viewer.opt.geomgroup[:] = 1
                print("\n[Respawn] Reset to spawn point.")
                print(f"\r[Command] vx: {cmd[0]:.2f}, vy: {cmd[1]:.2f}, wz: {cmd[2]:.2f}     ", end="")

            if lift_requested:
                _lift_robot(model, data, 0.1, last_action, cmd)
                lift_requested = False
                if viewer is not None:
                    viewer.opt.geomgroup[:] = 1
                print("\n[Lift] Raised robot by 0.10m.")
                print(f"\r[Command] vx: {cmd[0]:.2f}, vy: {cmd[1]:.2f}, wz: {cmd[2]:.2f}     ", end="")

            if step % sim_steps_per_ctrl == 0:
                obs = _build_obs(data, body_id, qpos_adr, dof_adr, default_qpos, last_action, cmd)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                with torch.inference_mode():
                    action = policy(obs_t).detach().cpu().numpy().squeeze(0)
                last_action = action.astype(np.float32)
                _apply_action(
                    model,
                    data,
                    leg_qpos_adr,
                    leg_dof_adr,
                    wheel_dof_adr,
                    default_leg_qpos,
                    last_action,
                    ctrl_cfg,
                )

            mujoco.mj_step(model, data)  # type: ignore[attr-defined]

            if viewer is not None:
                viewer.opt.geomgroup[:] = 1
                viewer.sync()

            step += 1
    finally:
        if viewer is not None:
            if viewer.is_running():
                viewer.close()
            time.sleep(0.2)


if __name__ == "__main__":
    main()
