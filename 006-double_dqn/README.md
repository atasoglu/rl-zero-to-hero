# 006 — Double DQN

> Fix DQN's overestimation bias by decoupling action selection from action evaluation across two networks.

## Overview

Standard DQN computes the TD target as `r + γ * max_a' Q_target(s', a')`. The same network both selects the best next action and evaluates it. Because `max` over noisy estimates is systematically larger than the true max, Q-values are consistently overestimated. This leads to suboptimal policies — the agent overestimates the value of certain state-action pairs and exploits them prematurely.

Double DQN (van Hasselt et al., 2015) separates these two roles: the **online network** selects the greedy action, but the **target network** evaluates it. This removes the upward bias while reusing the same two networks DQN already has.

## Algorithm

The only change from DQN is in the target computation:

```
# DQN target
target = r + γ * max_a' Q_target(s', a')

# Double DQN target
a* = argmax_a' Q_online(s', a')     # online net selects
target = r + γ * Q_target(s', a*)  # target net evaluates
```

Everything else — experience replay, target network sync, ε-greedy — stays identical to DQN.

## Environment

**LunarLander-v3** — 8D observation, 4 discrete actions. The shaped reward makes it sensitive to value overestimation: early overestimates lead the agent to fire thrusters unnecessarily, burning fuel and crashing. Double DQN's corrected targets produce more conservative, stable landing behaviour.

**CartPole-v1** is also supported as a fast sanity check.

## Key Takeaways

- Taking the max over noisy Q-estimates causes a systematic upward bias — DQN overestimates values
- Double DQN's fix is a one-line change in the target: select the action with the online net, evaluate with the target net
- The bias reduction leads to less overconfident policies, which is especially visible in shaped-reward environments like LunarLander
- Double DQN tends to outperform DQN on harder benchmarks while adding zero computational cost
- Overestimation bias is not unique to DQN — it resurfaces in actor-critic methods and is addressed again in TD3 (module 014)

## References

- van Hasselt et al. (2015) — [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- van Hasselt (2010) — [Double Q-Learning](https://proceedings.neurips.cc/paper/2010/hash/091d584fcea301d1b8fcea2b476b8d0d-Abstract.html) (original tabular version)
