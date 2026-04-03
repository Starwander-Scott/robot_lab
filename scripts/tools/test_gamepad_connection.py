#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Simple utility to verify that a gamepad is connected and reporting input."""

from __future__ import annotations

import argparse
import sys
import time

try:
    import pygame
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pygame is required: pip install pygame") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test gamepad connection and print live input values.")
    parser.add_argument("--gamepad-id", type=int, default=0, help="Joystick device index to open.")
    parser.add_argument("--poll-hz", type=float, default=20.0, help="Polling frequency for live state output.")
    return parser.parse_args()


def list_gamepads() -> int:
    count = pygame.joystick.get_count()
    print(f"Detected gamepads: {count}")
    for index in range(count):
        joystick = pygame.joystick.Joystick(index)
        joystick.init()
        print(
            f"[{index}] name={joystick.get_name()} axes={joystick.get_numaxes()} "
            f"buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}"
        )
        joystick.quit()
    return count


def format_state(joystick: pygame.joystick.Joystick) -> str:
    axes = [f"{joystick.get_axis(i):+.3f}" for i in range(joystick.get_numaxes())]
    buttons = [str(joystick.get_button(i)) for i in range(joystick.get_numbuttons())]
    hats = [str(joystick.get_hat(i)) for i in range(joystick.get_numhats())]
    return (
        f"axes=[{', '.join(axes)}] "
        f"buttons=[{', '.join(buttons)}] "
        f"hats=[{', '.join(hats)}]"
    )


def main() -> int:
    args = parse_args()

    pygame.init()
    pygame.joystick.init()

    gamepad_count = list_gamepads()
    if gamepad_count <= 0:
        print("No gamepad detected.")
        return 1
    if args.gamepad_id < 0 or args.gamepad_id >= gamepad_count:
        print(f"Invalid --gamepad-id={args.gamepad_id}. Available range: 0..{gamepad_count - 1}")
        return 1

    joystick = pygame.joystick.Joystick(args.gamepad_id)
    joystick.init()

    print(f"Opened gamepad [{args.gamepad_id}]: {joystick.get_name()}")
    print("Move sticks or press buttons. Press Ctrl+C to exit.")

    sleep_time = 1.0 / max(args.poll_hz, 1.0)

    try:
        while True:
            pygame.event.pump()
            if not joystick.get_init():
                print("Gamepad disconnected.")
                return 1
            print(format_state(joystick))
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nExit requested by user.")
        return 0
    finally:
        joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
