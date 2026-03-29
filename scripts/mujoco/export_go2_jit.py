# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Export a Go2 RSL-RL checkpoint to TorchScript actor for MuJoCo verification."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


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
        act = activations[activation]

        layers: list[nn.Module] = []
        dims = [obs_dim] + hidden_dims + [action_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Go2 actor to TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Path to RSL-RL model_XXXX.pt")
    parser.add_argument("--output", required=True, help="Output TorchScript path")
    parser.add_argument("--actor-obs-dim", type=int, default=45, help="Actor observation dim")
    parser.add_argument("--num-actions", type=int, default=12, help="Action dimension")
    parser.add_argument("--actor-hidden", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--activation", type=str, default="elu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    actor = ActorMLP(
        obs_dim=args.actor_obs_dim,
        hidden_dims=args.actor_hidden,
        action_dim=args.num_actions,
        activation=args.activation,
    )

    state = checkpoint["model_state_dict"]
    actor_state = {k: v for k, v in state.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    scripted_actor = torch.jit.script(actor)
    scripted_actor.save(args.output)

    print(f"Exported TorchScript actor to: {args.output}")


if __name__ == "__main__":
    main()
