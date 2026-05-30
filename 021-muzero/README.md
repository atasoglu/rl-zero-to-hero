# 021 — MuZero

> Plan with a learned model using MCTS — no hand-crafted rules, no access to environment dynamics.

## Overview

MuZero (Schrittwieser et al., 2019) extends AlphaZero to environments where the dynamics are unknown. It learns three networks jointly:

- **Representation**: `obs → hidden state h`
- **Dynamics**: `(h, a) → (next_h, reward)`
- **Prediction**: `h → (policy logits, value)`

At inference, **MCTS** uses these networks to simulate future trajectories and select actions. The MCTS visit counts become policy targets for training, creating a self-improving loop.

## Algorithm

```
select_action(obs):
    h = representation(obs)
    expand root node with prediction(h)
    for N simulations:
        selection: follow UCB scores to a leaf
        expansion: apply dynamics to get next h, reward
        prediction: get value + policy for new node
        backpropagation: update visit counts and values
    return action ∝ visit_counts^(1/τ)

update():
    sample batch from replay buffer
    h = representation(obs)
    a_onehot → dynamics(h, a) → next_h, pred_reward
    policy_logits, pred_value = prediction(h)
    loss = value_loss + reward_loss + policy_loss (cross-entropy with MCTS probs)
```

## Environment

**CartPole-v1** — chosen for fast iteration. MuZero's MCTS is O(simulations × depth), so complex environments require GPU acceleration; CartPole lets you observe convergence quickly.

Also supports: **LunarLander-v3**, **Acrobot-v1**.

## Key Takeaways

- MCTS converts a learned model into a strong planning algorithm; the model doesn't need to be perfect — just locally accurate
- Temperature annealing (τ: 1.0 → 0.1) makes early exploration stochastic and late training deterministic
- N-step returns (td_steps=5) give more informative value targets than 1-step TD
- UCB balances exploitation (high value) and exploration (high prior × low visit count)

## References

- Schrittwieser et al. (2019) — [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
