# 008 — Prioritized Experience Replay (PER)

> Sample transitions with higher TD error more frequently, correcting the bias with importance sampling weights.

## Overview

Uniform experience replay wastes compute on transitions the agent has already learned from well. PER (Schaul et al., 2015) assigns each transition a priority proportional to its TD error. Transitions the agent is surprised by (high TD error) are sampled more often. A sum tree data structure makes sampling and priority updates O(log n).

Changing the sampling distribution introduces a bias into the gradient estimate. **Importance sampling (IS) weights** `w = (N * P(i))^(-β)` correct for this by down-weighting high-priority samples. β is annealed from 0.4 to 1.0 over training so corrections are small early (when estimates are noisy) and exact by the end.

## Algorithm

```
# Proportional priority for transition i
P(i) = p_i^alpha / sum_j p_j^alpha
p_i  = |TD error_i| + epsilon

# IS correction weight
w_i = (N * P(i))^(-beta) / max_j w_j

# Weighted loss
loss = mean(w_i * (Q(s_i, a_i) - target_i)^2)

# After each update
p_i <- |new TD error_i| + epsilon
```

Uses a **sum tree** to maintain cumulative priorities for stratified O(log n) sampling.

## Environment

**LunarLander-v3** — default. Transitions near crashes have large TD errors and are replayed more. PER's benefit is most visible in environments with rare high-reward events.

**Atari** (ALE/Pong-v5, ALE/Breakout-v5, ALE/SpaceInvaders-v5) — the original benchmark from the paper, where PER showed the largest gains over uniform replay.

### Atari setup

```bash
pip install autorom
AutoROM --accept-license
```

## Key Takeaways

- Uniform replay wastes updates on transitions with near-zero TD error; PER focuses capacity on surprising transitions
- The sum tree enables O(log n) priority updates and O(log n) stratified sampling — crucial for large buffers
- IS weights correct the sampling bias; β=1 gives unbiased gradients, β<1 trades some bias for variance reduction
- Alpha controls how aggressively to prioritize (0=uniform, 1=fully greedy); 0.6 is a common default
- PER combines with any value-based algorithm (DQN, Double DQN, Dueling DQN)

## References

- Schaul et al. (2015) — [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
