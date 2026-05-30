# 007 — Dueling DQN

> Decompose Q(s, a) into state value V(s) and action advantage A(s, a) for better generalization across actions.

## Overview

In many states, the choice of action barely matters — the agent just needs to know how good the state is. Standard DQN must learn Q(s, a) independently for every action, even when actions are nearly equivalent. Dueling DQN (Wang et al., 2015) restructures the network to output two streams:

- **V(s)**: how good is this state, regardless of action
- **A(s, a)**: how much better is action a compared to the average

These are combined as `Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))`. Subtracting the mean advantage makes V and A identifiable (otherwise any constant could shift between them).

The key benefit: V(s) can be trained from every transition regardless of which action was taken, giving it more gradient signal and leading to better generalization.

## Algorithm

The only change from DQN is the network architecture:

```
# Standard QNetwork
Q(s, a) = MLP(s)[a]

# DuelingQNetwork
features = shared_layers(s)
V(s)    = value_head(features)          # scalar
A(s, a) = advantage_head(features)      # vector of size n_actions
Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
```

The training loop uses Double DQN-style targets (online net selects, target net evaluates).

## Environment

**LunarLander-v3** — default. The advantage of separating V from A is clearest in environments where many actions are near-equivalent for certain states (e.g., hovering at the right altitude).

**Atari** (ALE/Pong-v5, ALE/Breakout-v5, ALE/SpaceInvaders-v5) — the original benchmark from the paper. Uses a CNN encoder with frame stacking (4 frames of 84x84 grayscale). Requires Atari ROMs to be installed.

### Atari setup

```bash
# Install ROMs (AutoROM handles the download)
pip install autorom
AutoROM --accept-license
```

## Key Takeaways

- [ ] V(s) and A(s, a) decompose Q(s, a): V captures state quality, A captures action preference
- [ ] The mean-subtraction trick (`A - mean(A)`) ensures the decomposition is unique
- [ ] V(s) receives gradient from every transition, improving sample efficiency in states where action choice is irrelevant
- [ ] Dueling architecture is orthogonal to the training algorithm — it stacks with Double DQN, PER, and others
- [ ] For Atari, the shared CNN encoder with two small heads is the standard architecture

## References

- Wang et al. (2015) — [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
