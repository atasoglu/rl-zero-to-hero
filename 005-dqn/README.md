# 005 — Deep Q-Network (DQN)

> Off-policy value-based deep RL: replace the Q-table with a neural network, stabilize training with experience replay and a target network.

## Overview

Tabular Q-Learning breaks down when the state space is continuous or too large to enumerate. DQN (Mnih et al., 2015) approximates Q(s, a) with a neural network. Two tricks make the training stable:

**Experience replay** stores transitions `(s, a, r, s', done)` in a fixed-size buffer and samples random minibatches for gradient updates. This breaks the temporal correlations in sequential data that destabilize SGD.

**Target network** is a periodically-frozen copy of the online network used only for computing TD targets. Without it, the network chases a moving reference signal, causing oscillations or divergence.

## Algorithm

```
Initialize Q_net, target_net = copy(Q_net), replay buffer B
for each episode:
    obs = env.reset()
    while not done:
        action = ε-greedy(Q_net, obs)
        next_obs, reward, done = env.step(action)
        B.push(obs, action, reward, next_obs, done)
        if |B| >= batch_size:
            sample minibatch from B
            target = r + γ * max_a' Q_target(s', a') * (1 - done)
            loss = MSE(Q_net(s, a), target)
            gradient step on Q_net
        every target_update_freq steps: target_net <- Q_net
    decay ε
```

## Environment

**CartPole-v1** — 4D continuous observation (cart position, velocity, pole angle, angular velocity), 2 discrete actions. The episode ends when the pole falls or 500 steps are reached. A fast sanity check; solved in ~300-500 episodes.

**LunarLander-v2** — 8D observation, 4 discrete actions (noop, left, main, right thruster). Reward is shaped by landing safely between the flags. A more realistic benchmark for the full DQN pipeline; typically needs 600+ episodes.

## Key Takeaways

- A Q-table cannot scale to continuous observations; a neural network approximates Q(s, a) across the whole state space
- Experience replay breaks temporal correlations and allows reuse of past transitions
- The target network stabilizes training by decoupling the regression target from the online weights
- `target_update_freq` controls how often the target syncs; too low causes instability, too high slows learning
- DQN is still off-policy: the ε-greedy behavior policy and the greedy target policy are different
- Huber loss (or gradient clipping) can improve stability when TD errors are large; MSE works for these small envs

## References

- Mnih et al. (2013) — [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- Mnih et al. (2015) — [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 9 (function approximation)
