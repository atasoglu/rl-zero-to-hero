# 001 — Q-Learning

> Off-policy temporal difference control: learn the optimal policy by bootstrapping from the greedy value estimate.

## Overview

Q-Learning learns a state-action value function Q(s, a) directly without following the policy it is evaluating. At each step it uses the maximum Q-value of the next state as the update target, regardless of which action the agent actually takes next. This off-policy property lets it converge to the optimal Q-function even while exploring with an ε-greedy policy.

## Algorithm

```
Initialize Q(s, a) = 0 for all s, a
for each episode:
    s = env.reset()
    while not done:
        a = ε-greedy(Q, s)
        s', r, done = env.step(a)
        Q(s, a) += α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        s = s'
    decay ε
```

The update target `r + γ * max_a' Q(s', a')` is called the TD target. The difference between the target and the current estimate is the TD error (δ).

## Environment

**FrozenLake-v1** — a 4x4 grid where the agent navigates from start to goal without falling into holes. Sparse reward (+1 at goal), stochastic transitions. Good for observing convergence behaviour.

**Taxi-v3** — a 5x5 grid where a taxi must pick up and drop off a passenger. Shaped reward (penalty for illegal actions), larger state space (500 states). Good for observing how Q-Learning handles denser reward signals.

## Key Takeaways

- [ ] The Bellman optimality equation is the fixed point Q-Learning converges to
- [ ] Off-policy means the behavior policy (ε-greedy) and target policy (greedy) are different
- [ ] The TD error δ = r + γ max Q(s', a') - Q(s, a) is the learning signal
- [ ] ε-greedy balances exploration and exploitation; decaying ε shifts from explore to exploit
- [ ] Q-Learning converges to Q* given sufficient exploration and a decaying learning rate

## References

- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, University of Cambridge.
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 6.5
