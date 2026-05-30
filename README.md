# Reinforcement Learning: Zero to Hero

A hands-on RL curriculum from tabular methods to Vision-Language-Action models, aimed at practitioners working at the intersection of LLMs and Robotics.

Each module is a self-contained project with its own environment, runnable code, and a focused set of takeaways. The progression is deliberate: every tier builds on the intuitions from the previous one.

## About

This repository is a structured learning path through modern reinforcement learning. It starts with the fundamentals (Q-tables, Bellman equations) and ends with the techniques powering today's robot foundation models (RLHF, VLAs, π0-style policies).

**Target audience:** ML engineers and researchers who want to understand RL deeply, especially those interested in LLM alignment, robotic manipulation, or combining language and action.

## Prerequisites

- Python >= 3.10
- [`uv`](https://docs.astral.sh/uv/) as the package manager in every module
- Basic familiarity with Python, NumPy, and PyTorch (for Tier 2+)
- Linear algebra and probability fundamentals

## How to Use

Each module is an independent `uv` project. To run a module:

```bash
cd 001-q_learning
uv sync
uv run python src/main.py
```

Every module directory contains:

```
{id}-{name}/
├── README.md        # Background, algorithm walkthrough, Key Takeaways, references
├── pyproject.toml   # uv project with pinned dependencies
├── uv.lock          # Reproducible lock file
└── src/             # Implementation
```

## Curriculum

### Tier 1: Tabular RL

The foundations. No neural networks, just pure dynamic programming and temporal difference learning over discrete state spaces.

| # | Method | Environment | Key Concepts |
|---|--------|-------------|--------------|
| 001 | Q-Learning | FrozenLake, Taxi | Bellman equation, Q-table, ε-greedy |
| 002 | SARSA | FrozenLake, CliffWalking | On-policy vs off-policy |
| 003 | Monte Carlo Methods | Blackjack | Episode-based learning, first-visit MC |
| 004 | TD(λ) / Eligibility Traces | CartPole (discrete) | n-step returns, TD/MC unification |

### Tier 2: Deep RL

Function approximation with neural networks. The classic DQN paper and its successors, then policy gradient methods up to modern actor-critic algorithms.

| # | Method | Environment | Key Concepts |
|---|--------|-------------|--------------|
| 005 | DQN | CartPole, LunarLander | Experience replay, target network |
| 006 | Double DQN | LunarLander | Overestimation bias |
| 007 | Dueling DQN | Atari | Value / Advantage decomposition |
| 008 | Prioritized Experience Replay | Atari | TD-error based sampling |
| 009 | REINFORCE | CartPole | Policy gradient theorem |
| 010 | A2C / A3C | CartPole, MuJoCo | Advantage function, parallel actors |
| 011 | TRPO | MuJoCo | Trust region, KL constraint |
| 012 | PPO | MuJoCo, Humanoid | Clipped objective, practical workhorse |
| 013 | DDPG | Pendulum, HalfCheetah | Continuous actions, deterministic policy |
| 014 | TD3 | Ant, HalfCheetah | Double critic, delayed policy update |
| 015 | SAC | Humanoid | Entropy regularization, max-entropy RL |

### Tier 3: Advanced Methods

Model-based RL, exploration bonuses, and learned world models. These methods close the sample-efficiency gap with model-free approaches.

| # | Method | Environment | Key Concepts |
|---|--------|-------------|--------------|
| 016 | HER (Hindsight Experience Replay) | RobotFetch | Goal-conditioned RL, sparse rewards |
| 017 | Dyna-Q | GridWorld | Model-based + model-free hybrid |
| 018 | MBPO | MuJoCo | Model-based policy optimization, short rollouts, sample efficiency |
| 019 | World Models (Ha & Schmidhuber) | CarRacing | VAE + MDN-RNN, latent imagination |
| 020 | Dreamer v3 | Continuous control | RSSM, imagination-based training |
| 021 | MuZero | Atari, board games | Learned model + MCTS planning |
| 022 | Curiosity-Driven (ICM) | MontezumaRevenge | Intrinsic motivation, hard exploration |
| 023 | RND | Atari hard exploration | Random network distillation |

### Tier 4: Offline RL and Imitation Learning

Learning from fixed datasets and expert demonstrations without online environment interaction.

| # | Method | Dataset | Key Concepts |
|---|--------|---------|--------------|
| 024 | Behavioral Cloning | Minari / D4RL | Supervised imitation, covariate shift |
| 025 | IRL (Inverse RL) | GridWorld | Reward inference from demos |
| 026 | GAIL | MuJoCo | Adversarial imitation learning |
| 027 | CQL | Minari | Offline RL, OOD pessimism |
| 028 | IQL | Minari | Implicit Q-learning |
| 029 | Decision Transformer | Minari | RL as sequence modeling |
| 030 | Trajectory Transformer | Minari | Beam search planning with transformers |

### Tier 5: RLHF and LLM Alignment

Techniques for aligning language models with human preferences. The core pipeline behind InstructGPT, ChatGPT, and modern reasoning models.

| # | Method | Description | Key Concepts |
|---|--------|-------------|--------------|
| 031 | Reward Modeling | Human preference dataset | Bradley-Terry model, reward learning |
| 032 | PRM (Process Reward Model) | LLM reasoning | Step-level reward, reasoning chain supervision |
| 033 | RLHF (InstructGPT style) | LLM fine-tuning | RM + PPO pipeline end-to-end |
| 034 | DPO | LLM fine-tuning | Reward-free preference optimization |
| 035 | GRPO | LLM reasoning (DeepSeek-R1 style) | Group relative policy optimization |
| 036 | RLVR (Verifiable Rewards) | LLM math / code | Rule-based reward, outcome verification |
| 037 | Constitutional AI | LLM | Self-critique and revision loops |
| 038 | RLAIF | LLM | AI feedback as preference signal |

### Tier 6: Multi-Agent RL

Multiple agents learning simultaneously in cooperative, competitive, and mixed settings.

| # | Method | Environment | Key Concepts |
|---|--------|-------------|--------------|
| 039 | IQL (Independent Q-Learning) | PettingZoo | MARL baseline, non-stationarity |
| 040 | MAPPO | PettingZoo, SMAC | Multi-agent PPO, centralized value function |
| 041 | MADDPG | Cooperative / competitive | Centralized training, decentralized execution |
| 042 | QMIX | SMAC (StarCraft) | Cooperative MARL, monotonic mixing |
| 043 | Self-Play | Connect4, Chess | Competitive self-improvement, ELO |

### Tier 7: VLA and LLM + Robotics

Vision-Language-Action models and language-conditioned robot policies at the frontier where LLMs meet physical systems.

| # | Method | Description | Key Concepts |
|---|--------|-------------|--------------|
| 044 | Language-Conditioned RL | BabyAI, MiniGrid | NL instruction to policy |
| 045 | SayCan | LLM + robot primitives | Affordance scoring + LLM planning |
| 046 | RT-1 / RT-2 style | Simulated manipulation | Robotics transformer, VL pretraining |
| 047 | Code as Policy | LLM to executable code | Zero-shot robot control via code gen |
| 048 | ACT (Action Chunking with Transformers) | Simulated manipulation | Action chunking, temporal ensembling |
| 049 | Diffusion Policy | Robot manipulation | Denoising diffusion for action generation, multimodal distributions |
| 050 | OpenVLA | Open-source VLA | Vision-language-action fine-tuning |
| 051 | π0 (Pi-Zero) | Dexterous manipulation | Flow matching + VLA architecture |

## Module Structure Reference

Every module follows this layout:

```
{id}-{name}/
├── README.md
│   ├── Overview       - background and motivation
│   ├── Algorithm      - pseudocode or step-by-step walkthrough
│   ├── Environment    - which gym env and why
│   ├── Key Takeaways  - checklist of core concepts to internalize
│   └── References     - papers and resources
├── pyproject.toml
├── uv.lock
└── src/
```

### Dependencies by Tier

| Tier | Core dependencies |
|------|-------------------|
| 1: Tabular RL | `gymnasium`, `numpy`, `matplotlib` |
| 2: Deep RL | `gymnasium`, `torch`, `tensorboard`, `numpy` |
| 3: Advanced | `gymnasium[mujoco]`, `torch`, `tensorboard` |
| 4: Offline / Imitation | `gymnasium`, `torch`, `minari` |
| 5: RLHF / LLM | `transformers`, `trl`, `datasets`, `torch` |
| 6: MARL | `pettingzoo`, `gymnasium`, `torch` |
| 7: VLA / Robotics | `gymnasium-robotics`, `transformers`, `torch` |

All modules include `ipykernel` as a dev dependency for notebook exploration.
