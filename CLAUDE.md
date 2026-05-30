# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running a Module

Each module is an independent `uv` project. To run one:

```bash
cd 005-dqn
uv sync
uv run python src/train.py
```

Pass `--help` to see all hyperparameter flags (every module uses `argparse` with `ArgumentDefaultsHelpFormatter`):

```bash
uv run python src/train.py --help
```

Common flags available in every module:

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | varies | training episodes |
| `--watch N` | 5 | render trained agent for N episodes after training (0 to skip) |
| `--render-interval N` | 0 | render every N episodes during training |
| `--device auto\|cpu\|cuda\|mps` | auto | compute device (Tier 2+ only) |

## Architecture

### Shared Library: `common/`

All modules depend on `rl-common` (editable install from `../common`). It provides:

- `make_env(env_id, render_mode, seed, **kwargs)` — wraps any Gymnasium env with `RecordEpisodeStatistics`, seeded at 42
- `plot_rewards(rewards, title)` — plots a reward curve with a 100-episode rolling average
- `watch(run_fn, env, n_episodes)` — renders a trained agent; `run_fn` is a closure `(env) -> float`
- `ReplayBuffer(capacity, obs_dim, action_dim)` — numpy circular buffer used by off-policy Tier 2 agents

### Module Layout

Every module follows the same pattern:

```
{id}-{name}/
├── pyproject.toml      # uv project; depends on rl-common via editable path ../common
├── uv.lock
└── src/
    ├── train.py        # CLI entry point: parses args, builds env/agent, trains, evaluates, watches
    ├── agent.py        # Agent class (holds model, optimizer, buffer; exposes select_action / update)
    └── model.py        # Neural network definition (Tier 2+ only)
```

`train.py` always follows this call sequence:
1. `run_episode(agent, env, train=True/False)` — single episode loop
2. `train(agent, env, n_episodes, ...)` — calls `run_episode` in a tqdm loop
3. `evaluate(agent, env, n_episodes)` — greedy evaluation after training
4. `plot_rewards(...)` — save/show reward curve
5. `watch(...)` — optional visual playback

### Adding a New Module

1. Copy a nearby module directory as a template.
2. Update `pyproject.toml` (name, description, dependencies).
3. Keep the `rl-common` editable source reference: `rl-common = { path = "../common", editable = true }`.
4. Implement `agent.py` and, if needed, `model.py`; keep `train.py` structure consistent with existing modules.
5. Run `uv sync` before the first `uv run`.

### Extending `rl-common`

Add new utilities to `common/rl_common/`, export them from `common/rl_common/__init__.py`. Because all modules use an editable install, changes are picked up immediately without re-syncing.

## Module READMEs

Do **not** add a `## Usage` section to individual module READMEs. The root `README.md` already documents how to run any module (`uv sync` + `uv run python src/train.py`). Repeating it per-module is redundant.

## Code Style

### argparse

Every `add_argument` call must include a `help=` string. This is required for two reasons: it makes `--help` output self-documenting, and `ArgumentDefaultsHelpFormatter` only appends `(default: …)` to arguments that have a help string.

```python
# correct
p.add_argument("--lr", type=float, default=3e-4,
               help="AdamW learning rate")

# wrong — default value is hidden from --help output
p.add_argument("--lr", type=float, default=3e-4)
```
