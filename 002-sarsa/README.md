# 002 — SARSA

> On-policy temporal difference control: update Q-values using the action actually taken, not the greedy best.

## Overview

SARSA (State-Action-Reward-State-Action) is the on-policy counterpart to Q-Learning. The key difference is in the update target: instead of using `max Q(s', a')`, SARSA uses `Q(s', a')` where `a'` is the action the agent actually selects under its current ε-greedy policy. This makes SARSA more conservative — it accounts for the exploration noise in its own policy.

The practical consequence is visible on CliffWalking: Q-Learning finds the shortest path along the cliff edge (optimal but risky), while SARSA learns a safer path away from the cliff because it factors in the chance of accidentally stepping off during exploration.

## Algorithm

```
Initialize Q(s, a) = 0 for all s, a
for each episode:
    s = env.reset()
    a = ε-greedy(Q, s)
    while not done:
        s', r, done = env.step(a)
        a' = ε-greedy(Q, s')          # select next action now
        Q(s, a) += α * [r + γ * Q(s', a') - Q(s, a)]
        s, a = s', a'
    decay ε
```

The tuple (s, a, r, s', a') gives the algorithm its name.

## Environment

**FrozenLake-v1** — same 4x4 grid as in Q-Learning, used here to confirm that both methods converge on simple environments.

**CliffWalking-v0** — a 4x12 grid with a cliff along the bottom row. Large negative reward (-100) for stepping off. This environment is the canonical example for demonstrating the on-policy vs off-policy distinction: Q-Learning walks close to the cliff, SARSA avoids it.

## Key Takeaways

- [ ] SARSA is on-policy: the policy being evaluated and the policy generating experience are the same
- [ ] The update uses Q(s', a') where a' comes from the current ε-greedy policy, not from argmax
- [ ] On-policy methods are safer in environments with catastrophic states because they account for exploration risk
- [ ] Q-Learning converges to the optimal policy; SARSA converges to the best policy given the exploration strategy
- [ ] Both converge to the same policy as ε → 0, but during training they behave differently

## References

- Rummery, G. A., & Niranjan, M. (1994). *On-Line Q-Learning Using Connectionist Systems*. Technical report, Cambridge University.
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 6.4
