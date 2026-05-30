# 013 — Deep Deterministic Policy Gradient (DDPG)

> Extend DQN to continuous action spaces with a deterministic actor and Ornstein-Uhlenbeck exploration noise.

## Overview

DQN requires taking `argmax` over actions to select the greedy action, which is only tractable for discrete action spaces. DDPG (Lillicrap et al., 2015) handles continuous actions by learning a deterministic policy `μθ(s)` that directly outputs the action. A separate critic Q(s, a) evaluates it.

The actor is trained by maximizing the critic's output via the chain rule:

```
∇θ J ≈ E[ ∇a Q(s, a)|a=μθ(s) * ∇θ μθ(s) ]
```

DDPG borrows DQN's experience replay and target networks, now applied to both actor and critic. Target networks use **soft updates** (`τ * online + (1-τ) * target`) rather than periodic hard copies, which is smoother for continuous control.

Exploration is provided by **Ornstein-Uhlenbeck (OU) noise**, a correlated noise process that produces temporally smooth perturbations — better for physical control tasks than uncorrelated Gaussian noise.

## Algorithm

```
Initialize actor μθ, critic Qφ, target networks μθ', Qφ', replay buffer B
for each episode:
    reset OU noise process
    obs = env.reset()
    while not done:
        action = μθ(obs) + OU_noise
        next_obs, reward, done = env.step(action)
        B.push(obs, action, reward, next_obs, done)
        if |B| >= batch_size:
            sample minibatch
            y = r + γ * Qφ'(s', μθ'(s'))
            critic_loss = MSE(Qφ(s, a), y)
            actor_loss = -mean(Qφ(s, μθ(s)))
            update Qφ, μθ via gradient descent
            soft update: θ' <- τθ + (1-τ)θ', φ' <- τφ + (1-τ)φ'
```

## Environment

**Pendulum-v1** — 3D continuous observation, 1D continuous torque action. Classic continuous control benchmark; solved quickly with DDPG.

**MuJoCo** (HalfCheetah-v4, Ant-v4, Hopper-v4) — locomotion tasks with high-dimensional continuous actions. DDPG is sensitive to hyperparameters on these; TD3 (module 014) was designed to fix these instabilities.

### MuJoCo setup

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install gymnasium[mujoco]
```

## Key Takeaways

- [ ] The deterministic policy gradient theorem allows gradient-based actor optimization in continuous spaces
- [ ] Soft target updates (`τ=0.005`) are smoother than periodic hard copies — the target changes slowly every step
- [ ] OU noise produces correlated exploration; for many tasks, simple Gaussian noise works just as well
- [ ] The critic gradient flowing through Q(s, μ(s)) trains the actor — the actor never sees a reward directly
- [ ] DDPG is sensitive to hyperparameters and can suffer from Q-value overestimation — see TD3 (014) for the fix
- [ ] DDPG is off-policy, making experience replay efficient

## References

- Lillicrap et al. (2015) — [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
- Silver et al. (2014) — [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.html) (DPG theorem)
