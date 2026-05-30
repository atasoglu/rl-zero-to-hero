# 009 — REINFORCE

> Monte Carlo policy gradient: directly optimize the policy by following the gradient of expected return, estimated from full-episode rollouts.

## Overview

Q-Learning and DQN learn a value function and derive a policy from it. REINFORCE (Williams, 1992) takes the opposite approach: directly parameterize a stochastic policy πθ(a|s) and optimize θ to maximize expected return.

The policy gradient theorem gives the gradient direction:

```
∇θ J(θ) = E_π [ ∇θ log πθ(a|s) * G_t ]
```

In practice, the agent collects a full episode, computes discounted returns G_t from each timestep, and uses `-(log π * G_t)` as the loss (negative because we maximize return, not minimize loss).

**Return normalization** — standardizing G_t per episode stabilizes gradients when returns vary widely across episodes.

## Algorithm

```
Initialize policy network πθ
for each episode:
    rollout: collect (s_t, a_t, r_t) until done
    compute returns G_t = sum_{k=t}^T γ^(k-t) * r_k
    normalize G_t: (G_t - mean) / std
    loss = -mean(log πθ(a_t | s_t) * G_t)
    gradient step on θ
```

No replay buffer, no target network. One gradient update per episode.

## Environment

**CartPole-v1** — 4D continuous observation, 2 discrete actions. REINFORCE converges more slowly than DQN but demonstrates the core policy gradient idea with minimal machinery. Typically needs 500-1000 episodes.

**LunarLander-v3** is also supported but requires many more episodes due to high variance.

## Key Takeaways

- [ ] Policy gradient directly optimizes πθ — no need to learn a Q-function first
- [ ] `∇θ log π(a|s) * G` is the policy gradient: actions that led to high return are made more probable
- [ ] REINFORCE is high variance because G_t is estimated from a single trajectory; this is why it needs many episodes
- [ ] Return normalization (standardize per episode) reduces variance and stabilizes training
- [ ] REINFORCE is on-policy: the policy that collected the episode must be the same policy being updated
- [ ] The baseline trick (subtracting a state-dependent baseline from G_t) reduces variance without bias — A2C (module 010) formalizes this

## References

- Williams (1992) — [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 13
