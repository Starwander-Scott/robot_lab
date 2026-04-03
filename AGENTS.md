# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `source/robot_lab/robot_lab`, with task definitions split between `tasks/direct` and `tasks/manager_based`. Package metadata and install logic live in `source/robot_lab/config`, `source/robot_lab/setup.py`, and `source/robot_lab/pyproject.toml`. Runtime scripts are under `scripts/`, mainly `scripts/reinforcement_learning/` for train/play entrypoints and `scripts/tools/` for utilities such as `list_envs.py`. Robot data and large static assets are stored in `source/robot_lab/data/` and `assets/`. Docs and screenshots live in `docs/`; container files live in `docker/`.

## Build, Test, and Development Commands
Install the package into the Isaac Lab Python environment with `python -m pip install -e source/robot_lab`. Verify the extension is visible with `python scripts/tools/list_envs.py`. Typical local runs are `python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless` and `python scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK_NAME>`. Run repository checks with `pre-commit run --all-files`; this applies Ruff linting, Ruff formatting, codespell, and file hygiene hooks.

## Coding Style & Naming Conventions
Use Python 3.10+ compatible code; the repo targets Python 3.11 for type checking. Ruff enforces formatting, import order, and a 120-character line limit, so prefer `pre-commit` over manual formatting. Follow existing naming patterns: snake_case for modules and functions, `*_cfg.py` for configuration modules, and task-specific directories grouped by robot family. Keep imports aligned with the custom Isaac Lab import sections already configured in `pyproject.toml`.

## Testing Guidelines
There is no dedicated first-party `tests/` directory in this repository. Treat validation as a mix of static checks and task smoke tests: run `pre-commit run --all-files`, then execute the relevant `train.py`, `play.py`, or utility script for the task you changed. If you add automated tests later, place them in a top-level `tests/` directory and use `test_*.py` naming so they fit standard `pytest` discovery.

## Commit & Pull Request Guidelines
Recent history mixes brief fixes with Conventional Commit-style messages such as `fix(cusrl): ...` and `feat: ...`; prefer the conventional form for new work. Keep commit subjects imperative and scoped when useful. Pull requests should summarize the change, note the linked issue, list any dependency impact, and confirm `pre-commit run --all-files` passes. Include screenshots only for UI or visualization changes, and add your name to `CONTRIBUTORS.md` if required by the PR template.
