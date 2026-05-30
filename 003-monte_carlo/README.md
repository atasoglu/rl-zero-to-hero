# 003 — Monte Carlo Methods

> Learn from complete episodes: no bootstrapping, no model, just averaging actual returns.

## Overview

Monte Carlo (MC) methods learn directly from sampled episodes. Unlike TD methods (Q-Learning, SARSA), MC waits until the end of an episode to update its estimates. The value of a state is estimated as the average of the actual returns observed after visiting that state.

This has a fundamental tradeoff with TD methods:
- MC has high variance (returns are noisy) but zero bias (no bootstrapping approximation)
- TD has low variance but introduces bias through the bootstrap target

Blackjack is a natural fit for MC: episodes are short, the game is episodic by definition, and there is no known model of the environment.

## Algorithm

First-visit MC control with ε-greedy policy:

```
Initialize Q(s, a) = 0, Returns(s, a) = []
for each episode:
    generate episode: s0, a0, r1, s1, a1, r2, ..., sT
    G = 0
    for t = T-1 downto 0:
        G = r_{t+1} + γ * G
        if (s_t, a_t) not seen earlier in episode:  # first-visit
            Returns(s_t, a_t).append(G)
            Q(s_t, a_t) = mean(Returns(s_t, a_t))
    update ε-greedy policy from Q
```

Implementation uses an incremental mean to avoid storing all returns.

## Environment

**Blackjack-v1** — the card game. State is a tuple of (player sum, dealer showing card, usable ace). Action space is {hit, stick}. Reward is +1 win, -1 loss, 0 draw. Episodes are short (a few steps) and the state space is small enough to represent as a dict.

## Key Takeaways

- [ ] MC methods require complete episodes — they cannot be applied to continuing (non-episodic) tasks
- [ ] First-visit MC: only the first occurrence of a (s, a) pair per episode is used for the update
- [ ] No bootstrapping means no bias, but high variance — many episodes are needed
- [ ] The incremental mean update `Q += (G - Q) / n` avoids storing all past returns
- [ ] MC converges to the true Q-values under the policy being followed (on-policy)

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 5
