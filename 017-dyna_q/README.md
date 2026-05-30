# 017 — Dyna-Q

> Model-free Q-learning augmented with a learned tabular world model; plan from simulated experience to improve sample efficiency.

## Overview

Dyna-Q (Sutton, 1991) bridges model-free and model-based RL. After each real environment step, the agent:

1. Updates the Q-table with the real transition (standard Q-learning).
2. Updates a learned **world model** (tabular: maps `(s, a) → {(s', r)}` from observed counts).
3. Performs `n_planning` additional Q-learning updates using **simulated transitions** sampled from the world model.

The key insight: simulated experience is cheap. With an accurate model, many planning steps can be done per real step, dramatically accelerating convergence.

## Algorithm

```
Initialize Q-table, empty model M
for each step:
    a = ε-greedy(Q, s)
    observe s', r
    Q(s, a) += α * (r + γ * max_a' Q(s', a') - Q(s, a))   # direct RL
    M[s, a].add(s', r)                                      # model update

    for n in range(n_planning):                             # planning
        (s_sim, a_sim) = random previously-seen pair
        (s'_sim, r_sim) = M[s_sim, a_sim].sample()
        Q(s_sim, a_sim) += α * (r_sim + γ * max Q(s'_sim, ·) - Q(s_sim, a_sim))
```

## Environment

**FrozenLake-v1** (4×4, deterministic) — small enough that the tabular model fits in memory exactly, making the planning effect clearly visible.

Compare with `--n-planning 0` (pure Q-learning) to see how faster Dyna-Q converges.

## Key Takeaways

- Planning from a learned model replaces real environment interactions — cheaper than collecting more data
- With a perfect model, `n_planning` steps of planning ≈ `n_planning` additional real steps
- Model errors accumulate over long rollouts; Dyna-Q uses 1-step simulated transitions, keeping compounding error minimal
- The model is only ever sampled at previously observed `(s, a)` pairs — it never extrapolates

## References

- Sutton (1991) — [Dyna, an Integrated Architecture for Learning, Planning, and Reacting](https://dl.acm.org/doi/10.1145/122344.122377)
- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*, Chapter 8
