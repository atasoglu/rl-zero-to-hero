# 004 — TD(λ) / Eligibility Traces

> Bridge between TD and Monte Carlo: propagate credit backwards through an episode using eligibility traces.

## Overview

TD(λ) unifies the TD and MC families through a single parameter λ ∈ [0, 1]. At λ=0 it reduces to one-step TD (SARSA). At λ=1 it approaches Monte Carlo. Values in between give a weighted mix of n-step returns, with recent transitions receiving more credit.

The mechanism is eligibility traces: a memory variable `e(s, a)` that tracks how recently and frequently each state-action pair was visited. When a TD error is observed, it is broadcast back to all recently visited pairs weighted by their trace value. This solves the credit assignment problem more efficiently than waiting for episode end or manually computing n-step returns.

## Algorithm

SARSA(λ) with accumulating traces:

```
Initialize Q(s, a) = 0, e(s, a) = 0 for all s, a
for each episode:
    reset traces: e = 0
    s = env.reset(), a = ε-greedy(Q, s)
    while not done:
        s', r, done = env.step(a)
        a' = ε-greedy(Q, s')
        δ = r + γ * Q(s', a') - Q(s, a)
        e(s, a) += 1                    # accumulating trace
        Q += α * δ * e                  # update all pairs
        e *= γ * λ                      # decay all traces
        s, a = s', a'
    decay ε
```

CartPole has continuous observations, so they are discretized into bins before indexing the Q-table.

## Environment

**CartPole-v1** — a pole balanced on a cart. Observation: cart position, cart velocity, pole angle, pole angular velocity (all continuous). Action: push left or right. Reward: +1 per timestep. Episode ends when pole falls or cart goes out of bounds. Max episode length is 500.

The continuous observations are discretized into 10 bins per dimension using fixed bin edges, producing a ~10,000-state discrete space.

## Key Takeaways

- [ ] λ interpolates between TD(0) (one-step bootstrap) and MC (full return): λ=0 is SARSA, λ=1 ≈ MC
- [ ] Eligibility traces solve credit assignment by propagating the TD error backward in time
- [ ] Accumulating traces: e(s, a) += 1 each visit; replacing traces: e(s, a) = 1 (avoids boosting frequently visited states)
- [ ] Traces decay by γλ each step; pairs not recently visited receive small updates
- [ ] Discretization is a simple way to apply tabular methods to continuous spaces, but does not scale

## References

- Sutton, R. S. (1988). *Learning to Predict by the Methods of Temporal Differences*. Machine Learning.
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 12
