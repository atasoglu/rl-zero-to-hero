# 015 — Soft Actor-Critic (SAC)

> Maximum-entropy RL: optimize for both reward and policy entropy simultaneously, with automatic temperature tuning.

## Overview

TD3 stabilizes DDPG with engineering fixes. SAC (Haarnoja et al., 2018) takes a principled approach: add entropy to the reward signal as a regularizer. The agent is rewarded not just for accumulating returns, but for maintaining a high-entropy (diverse) policy.

The modified objective is:

```
J(π) = E[ sum_t γ^t (r_t + α * H(π(· | s_t))) ]
```

This encourages exploration naturally, prevents premature convergence to suboptimal deterministic policies, and makes SAC one of the most stable and sample-efficient off-policy algorithms available.

**Automatic temperature tuning** adjusts α dynamically to maintain a target entropy level, removing the need to hand-tune α. The target entropy is typically set to `-action_dim`.

SAC uses a **stochastic actor** with reparameterized sampling (mean + std from the network, sampled via the reparameterization trick). This allows gradients to flow through the action sample into the policy.

## Algorithm

```
Initialize actor π, twin critics Q1/Q2, target critics Q1'/Q2', temperature α
for each step:
    action, log_prob = actor.sample(obs)   # reparameterized
    push to replay buffer

    sample minibatch
    # Critic update (entropy-regularized Bellman)
    y = r + γ * (min(Q1', Q2')(s', a') - α * log_π(a'|s'))
    minimize MSE(Q1(s,a), y) + MSE(Q2(s,a), y)

    # Actor update
    maximize E[ min(Q1, Q2)(s, a) - α * log_π(a|s) ]

    # Alpha update (auto-tune)
    minimize -α * (log_π(a|s) + target_entropy)

    soft update target critics
```

## Environment

**Pendulum-v1** — quick check; SAC converges in ~100-200 episodes.

**MuJoCo** (HalfCheetah-v4, Hopper-v4, Walker2d-v4, Humanoid-v4, Ant-v4) — SAC is competitive with or better than TD3 on all standard locomotion benchmarks, especially with auto alpha tuning.

### MuJoCo setup

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install gymnasium[mujoco]
```

## Key Takeaways

- [ ] Maximum-entropy RL adds entropy H(π) to the reward: the agent is incentivized to be uncertain between equally good actions
- [ ] Entropy regularization provides implicit exploration without needing a separate noise process
- [ ] The reparameterization trick (`a = tanh(μ + σ * ε), ε ~ N(0,1)`) allows gradients to flow through the stochastic action
- [ ] Auto alpha tuning adjusts temperature to maintain `target_entropy = -action_dim` — one fewer hyperparameter
- [ ] SAC with twin critics removes overestimation bias; the entropy term prevents Q-value underestimation too
- [ ] SAC is currently the state-of-the-art off-policy algorithm for continuous control and a strong baseline for robot learning research

## References

- Haarnoja et al. (2018) — [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- Haarnoja et al. (2018) — [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) (auto alpha version)
