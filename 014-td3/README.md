# 014 — Twin Delayed Deep Deterministic Policy Gradient (TD3)

> Fix DDPG's instability with three targeted interventions: twin critics, delayed policy updates, and target policy smoothing.

## Overview

DDPG is powerful but brittle. It tends to overestimate Q-values (inherited from DQN) and is sensitive to hyperparameters. TD3 (Fujimoto et al., 2018) diagnoses three failure modes and addresses each with a separate technique:

1. **Twin critics** (the "double" in the name) — train two independent critics Q1, Q2 and use `min(Q1, Q2)` as the target. This is the Double DQN idea applied to continuous actions, removing overestimation.

2. **Delayed policy updates** — update the actor only every `policy_delay` critic steps. Since the critic is used to train the actor, a noisy critic produces noisy actor updates. Waiting for the critic to stabilize first produces more reliable policy gradients.

3. **Target policy smoothing** — add small clipped noise to the target action when computing the Bellman target. This prevents the critic from fitting sharp peaks around deterministic actions and reduces variance.

## Algorithm

```
# Critic update (every step)
noise = clip(N(0, σ), -c, c)
next_a = clip(actor_target(s') + noise, -scale, scale)
y = r + γ * min(Q1_target(s', next_a), Q2_target(s', next_a))
critic_loss = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)

# Actor and target update (every policy_delay steps)
actor_loss = -mean(Q1(s, actor(s)))
soft update: actor_target, critic_target
```

## Environment

**Pendulum-v1** — simplest continuous control; converges quickly.

**MuJoCo** (HalfCheetah-v4, Ant-v4, Hopper-v4, Walker2d-v4) — TD3's primary benchmark. Consistently outperforms DDPG on locomotion tasks.

### MuJoCo setup

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install gymnasium[mujoco]
```

## Key Takeaways

- [ ] Twin critics (`min(Q1, Q2)`) address overestimation bias — the same motivation as Double DQN but in continuous action space
- [ ] Delayed actor updates decouple policy learning from noisy early critic estimates
- [ ] Target policy smoothing regularizes the value function: adding noise to target actions prevents sharp Q-value peaks
- [ ] The three tricks in TD3 each address a distinct failure mode; removing any one of them degrades performance
- [ ] TD3's exploration noise is simple Gaussian (not OU) — this is simpler and works equally well in practice
- [ ] SAC (module 015) provides a principled alternative: entropy maximization rather than heuristic noise

## References

- Fujimoto et al. (2018) — [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
