# 022 — ICM (Intrinsic Curiosity Module)

> Add curiosity-driven intrinsic rewards to PPO using forward and inverse dynamics models, enabling exploration in sparse-reward environments.

## Overview

Hard-exploration environments like Montezuma's Revenge have so few extrinsic rewards that standard PPO learns nothing. ICM (Pathak et al., 2017) provides a dense intrinsic reward: the agent is rewarded for visiting states that its own forward model finds surprising.

The ICM has two components sharing a CNN encoder φ:

- **Inverse model**: `(φ(s), φ(s')) → predicted action` — trained to identify what action caused a transition. This forces φ to capture action-controllable features, ignoring irrelevant background noise.
- **Forward model**: `(φ(s), a) → predicted φ(s')` — predicts the next feature. High prediction error = high novelty = high intrinsic reward.

## Algorithm

```
for each rollout of n_steps:
    collect (obs, action, ext_reward, next_obs) with PPO policy

    intrinsic_reward = η * ||forward(φ(s), a) - φ(s')||²

    total_reward = ext_reward + intrinsic_reward

    PPO update with total_reward
    ICM update:
        inverse_loss = cross_entropy(predicted_action, actual_action)
        forward_loss = ||predicted_φ(s') - φ(s')||²
        icm_loss = (1 - β) * inverse_loss + β * forward_loss
```

## Architecture

| Component | Input | Output |
|-----------|-------|--------|
| Encoder φ | (4, 84, 84) frame | 512-dim features |
| Inverse model | (φ(s), φ(s')) | action logits |
| Forward model | (φ(s), a_onehot) | predicted φ(s') |
| Actor-critic | (4, 84, 84) | action dist + value |

## Environment

**ALE/MontezumaRevenge-v5** — the canonical hard-exploration benchmark. With pure PPO: ~0 reward. With ICM: agent actively explores new rooms.

Also supports: **ALE/Pitfall-v5**, **ALE/PrivateEye-v5**.

## Key Takeaways

- The inverse model filters out environment noise (e.g. TV static) that the agent can't control — only action-relevant features survive in φ
- Intrinsic reward is the model's prediction error, not hand-crafted — it's zero once the agent has seen a transition enough times
- β controls forward vs inverse loss balance; λ controls ICM vs PPO loss balance
- ICM is compatible with any on-policy algorithm (PPO is used here)

## References

- Pathak et al. (2017) — [Curiosity-Driven Exploration by Self-Supervised Prediction](https://arxiv.org/abs/1705.05363)
