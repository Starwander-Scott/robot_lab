# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Convert a GO2 flat-trained checkpoint so it can be resumed in the rough environment.

Background
----------
GO2 flat env and GO2 rough env share the same **actor** observation space (45-dim,
proprioceptive-only; both configs set ``observations.policy.height_scan = None``).
The difference is in the **critic** observation space:

  - Flat critic: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
                 + velocity_commands(3) + joint_pos(12) + joint_vel(12)
                 + actions(12) = 48 dims
  - Rough critic: same 48 dims + height_scan(160) appended at end = 208 dims

This script extends the critic's first linear layer weight matrix from [512, 48]
to [512, 208] by appending 160 zero-initialised columns.  The actor is copied
unchanged.  The iteration counter is reset to 0 so the terrain curriculum starts
from the beginning.

Usage
-----
    python scripts/tools/convert_flat_to_rough_go2.py \\
        logs/rsl_rl/unitree_go2_flat/2026-04-02_11-49-52/model_71000.pt \\
        logs/rsl_rl/unitree_go2_rough/from_flat_71000/model_rough_init.pt
"""

import argparse
import sys
from pathlib import Path

import torch


def convert(flat_ckpt_path: str, out_path: str, height_scan_dim: int = 187) -> None:
    """Extend critic input layer to accommodate height_scan observations.

    Args:
        flat_ckpt_path: Path to the source flat-terrain checkpoint.
        out_path: Destination path for the converted checkpoint.
        height_scan_dim: Number of height-scan dimensions added in rough env.
            GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) generates 17×11=187 points
            (positions: -0.8 to +0.8 step 0.1 = 17; -0.5 to +0.5 step 0.1 = 11).
    """
    src = Path(flat_ckpt_path)
    dst = Path(out_path)

    if not src.exists():
        print(f"[ERROR] Source checkpoint not found: {src}")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading flat checkpoint: {src}")
    ckpt = torch.load(str(src), map_location="cpu")

    state: dict = ckpt["model_state_dict"]

    # ------------------------------------------------------------------ #
    # Actor: observation space is identical in flat and rough for GO2.    #
    # No modification needed.                                             #
    # ------------------------------------------------------------------ #
    actor_in = state["actor.0.weight"].shape[1]
    print(f"[INFO] Actor input dim: {actor_in}  (unchanged)")

    # ------------------------------------------------------------------ #
    # Critic: rough env appends height_scan at the END of the critic obs. #
    # Extend critic.0.weight columns by zero-padding.                     #
    # ------------------------------------------------------------------ #
    critic_w = state["critic.0.weight"]          # [hidden, obs_dim_flat]
    hidden_dim, obs_dim_flat = critic_w.shape
    obs_dim_rough = obs_dim_flat + height_scan_dim

    zero_pad = torch.zeros(hidden_dim, height_scan_dim, dtype=critic_w.dtype)
    state["critic.0.weight"] = torch.cat([critic_w, zero_pad], dim=1)

    print(
        f"[INFO] Critic first-layer extended: "
        f"[{hidden_dim}, {obs_dim_flat}] → [{hidden_dim}, {obs_dim_rough}]"
    )
    print(f"[INFO] Zero-padded {height_scan_dim} columns for height_scan dims")

    # ------------------------------------------------------------------ #
    # Reset optimizer state — Adam momentum buffers (exp_avg,             #
    # exp_avg_sq) still have the old critic shape and will cause a        #
    # size-mismatch error during optimizer.step().  Clearing the 'state'  #
    # dict lets Adam reinitialise cleanly while preserving param_groups   #
    # (learning rate, betas, etc.).  RSL-RL's load() always accesses      #
    # loaded_dict["optimizer_state_dict"], so the key must remain.        #
    # ------------------------------------------------------------------ #
    if "optimizer_state_dict" in ckpt:
        ckpt["optimizer_state_dict"]["state"] = {}
        print("[INFO] Optimizer state cleared (incompatible momentum shapes)")

    # ------------------------------------------------------------------ #
    # Reset iteration counter so terrain curriculum starts from 0.        #
    # ------------------------------------------------------------------ #
    original_iter = ckpt.get("iter", "N/A")
    ckpt["iter"] = 0
    print(f"[INFO] Iteration counter reset: {original_iter} → 0")

    torch.save(ckpt, str(dst))
    print(f"[INFO] Converted checkpoint saved to: {dst}")
    print()
    print("[INFO] Resume rough training with:")
    print(
        "  python scripts/reinforcement_learning/rsl_rl/train.py \\\n"
        "    --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \\\n"
        "    --headless \\\n"
        "    --resume \\\n"
        f"    --load_run from_flat_71000 \\\n"
        f"    --checkpoint model_rough_init.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GO2 flat checkpoint for rough-terrain training.")
    parser.add_argument("flat_ckpt", type=str, help="Path to source flat checkpoint (.pt)")
    parser.add_argument("out_path", type=str, help="Destination path for converted checkpoint (.pt)")
    parser.add_argument(
        "--height_scan_dim",
        type=int,
        default=187,
        help="Number of height-scan dims added by rough env (default: 187 = 17×11 grid, resolution=0.1, size=[1.6,1.0])",
    )
    args = parser.parse_args()
    convert(args.flat_ckpt, args.out_path, args.height_scan_dim)
