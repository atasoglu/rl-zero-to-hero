# 023 — RND (Random Network Distillation)

> Measure state novelty by how well a trainable predictor matches a fixed random network — novel states are harder to predict.

## Overview

RND (Burda et al., 2018) is a simpler and more scalable alternative to ICM. Instead of learning a forward dynamics model, it uses a fixed randomly-initialized target network as a "novelty oracle":

- **Target network** (frozen): random CNN that maps `obs → fixed_features`
- **Predictor network** (trained): learns to match the target's output

For states seen many times, the predictor has had many updates on those observations → small error. For novel states, the predictor is unprepared → large error. This error becomes the intrinsic reward.

**Two value heads**: separate critics for extrinsic (episodic) and intrinsic (non-episodic) rewards are maintained. Intrinsic rewards are non-episodic because curiosity is a life-long signal, not reset at episode boundaries.

## Algorithm

```
for each rollout:
    collect (obs, action, ext_reward) with PPO

    # RND intrinsic reward
    r_int = ||predictor(obs) - target(obs)||²  (target is frozen)
    r_int = normalize(r_int)  # running mean/std

    # Separate GAE for ext (episodic) and int (non-episodic)
    total_advantage = ext_coef * adv_ext + int_coef * adv_int

    PPO update with combined advantage
    predictor update: minimize ||predictor(obs) - target(obs)||²
```

## Environment

**ALE/MontezumaRevenge-v5** — same hard-exploration benchmark as ICM (022). RND typically outperforms ICM here due to its simpler and more stable training dynamics.

## Key Takeaways

- [ ] No dynamics model needed: novelty comes from distilling a random function, not predicting transitions
- [ ] The fixed target provides a stable reference signal — no moving target problem
- [ ] Non-episodic intrinsic rewards accumulate across episodes, rewarding long-term exploration
- [ ] Running mean/std normalization of intrinsic rewards is critical for stable training
- [ ] RND with PPO matches or exceeds ICM at a fraction of the implementation complexity

## References

- Burda et al. (2018) — [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
