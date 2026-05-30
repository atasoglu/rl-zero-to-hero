# 012 — Proximal Policy Optimization (PPO)

> Approximate TRPO's trust region with a simple clipped surrogate objective — same stability, far less complexity.

## Overview

TRPO constrains policy updates with a KL bound enforced via second-order optimization. PPO (Schulman et al., 2017) achieves similar stability with a first-order trick: the **clipped surrogate objective** directly penalizes the ratio `r = π_new / π_old` when it moves too far from 1.

```
L_CLIP = E[ min(r * A, clip(r, 1-ε, 1+ε) * A) ]
```

If the ratio exceeds the clip range (1-ε, 1+ε), the gradient of that term goes to zero — the update is simply ignored for that sample. In practice, PPO can reuse each trajectory for multiple gradient epochs, making it more sample-efficient than vanilla policy gradient while remaining simple to implement.

PPO-Clip has become the dominant practical workhorse algorithm in RL research and RLHF fine-tuning (it is the policy optimizer in InstructGPT and most RLHF pipelines).

## Algorithm

```
for each iteration:
    collect K episodes with current policy πθ_old
    compute advantages with GAE

    for n_epochs:
        for minibatch in shuffled(data):
            r = exp(log π_θ(a|s) - log π_θ_old(a|s))
            L_CLIP = min(r * A, clip(r, 1-ε, 1+ε) * A)
            L_VF   = (V_θ(s) - G_t)^2
            L_ENT  = H(π_θ(·|s))
            loss = -L_CLIP + c_v * L_VF - c_e * L_ENT
            gradient step
```

## Environment

**CartPole-v1** — solves quickly, good for inspecting the clipping mechanism.

**LunarLander-v3** — demonstrates PPO's ability to handle sparse and shaped rewards.

**MuJoCo** (HalfCheetah-v4, Hopper-v4, Walker2d-v4, Humanoid-v4) — continuous control benchmark. PPO is competitive with DDPG/TD3/SAC on locomotion tasks.

### MuJoCo setup

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install gymnasium[mujoco]
```

## Key Takeaways

- [ ] The clipped objective is a first-order proxy for TRPO's KL constraint — simpler but nearly as stable
- [ ] `clip_eps=0.2` means the ratio r is clamped to [0.8, 1.2]; updates that would push r outside this range are suppressed
- [ ] Reusing data for multiple epochs (`n_epochs`) increases sample efficiency — the clipping prevents exploitation of stale data
- [ ] GAE advantage estimates reduce variance; `gae_lambda=0.95` is a strong default
- [ ] PPO is on-policy but can reuse recent data for a few epochs with the clip protecting against too-large updates
- [ ] PPO is the policy optimizer used in RLHF pipelines (InstructGPT, ChatGPT); understanding it is essential for Tier 5

## References

- Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Ouyang et al. (2022) — [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT — PPO in RLHF)
