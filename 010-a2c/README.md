# 010 — Advantage Actor-Critic (A2C)

> Reduce REINFORCE's high variance by replacing raw returns with advantage estimates from a learned value baseline.

## Overview

REINFORCE works but suffers from high variance: returns from different episodes can differ wildly, causing noisy gradient updates. A2C introduces a **critic** that learns the state value V(s) and uses it as a baseline. The policy (actor) is updated using the **advantage** A(s, a) = Q(s, a) - V(s), which measures how much better a specific action is compared to the average action in that state.

This combination reduces variance without introducing bias (since subtracting a baseline dependent only on state doesn't change the expected gradient). Gradient clipping and entropy regularization further stabilize training.

The "A3C" variant runs multiple actors in parallel asynchronously; A2C is the synchronous version that collects from all actors before each update. This implementation is single-actor A2C.

## Algorithm

```
Initialize shared ActorCritic network (actor head + critic head)
for each episode:
    rollout: collect (s_t, a_t, r_t) until done
    compute returns G_t = sum_{k=t}^T γ^(k-t) * r_k
    advantage A_t = G_t - V(s_t)  (critic as baseline)

    actor_loss  = -mean(log π(a_t | s_t) * A_t)
    critic_loss = mean((G_t - V(s_t))^2)
    entropy_bonus = -mean(H(π(· | s_t)))  (encourages exploration)
    loss = actor_loss + c_v * critic_loss + c_e * entropy_bonus

    gradient step (with gradient clipping)
```

## Environment

**CartPole-v1** — fast feedback, good for understanding actor-critic dynamics.

**LunarLander-v3** — more complex shaped reward. A2C's lower variance pays off here.

**MuJoCo** (HalfCheetah-v4, Hopper-v4, Walker2d-v4) — continuous action spaces. Note: this implementation uses a discrete action wrapper; for true continuous control see DDPG (013) or PPO (012).

### MuJoCo setup

```bash
# Install system dependency
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# MuJoCo is included in gymnasium[mujoco]
pip install gymnasium[mujoco]
```

## Key Takeaways

- The critic provides a baseline V(s) that reduces variance without biasing the gradient
- Advantage A(s, a) = G_t - V(s_t) measures relative action quality: positive means better than average, negative means worse
- Actor and critic share a feature backbone — this improves sample efficiency but can cause conflicting gradients
- Entropy regularization prevents the policy from collapsing to a deterministic one too early
- Gradient clipping prevents catastrophic updates when advantage estimates are noisy
- A2C is on-policy: collected episodes must come from the current policy

## References

- Mnih et al. (2016) — [Asynchronous Methods for Deep Reinforcement Learning (A3C)](https://arxiv.org/abs/1602.01783)
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 13.5 (actor-critic methods)
