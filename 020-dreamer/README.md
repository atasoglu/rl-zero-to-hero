# 020 — Dreamer (Simplified DreamerV3)

> Learn a recurrent latent world model (RSSM) and train actor-critic entirely through latent imagination — no gradients through the real environment.

## Overview

Dreamer (Hafner et al., 2019–2023) is the state-of-the-art model-based RL algorithm for continuous control. All policy training happens inside the **world model's imagination**: the actor never touches the real environment during learning.

The core innovation is the **RSSM** (Recurrent State Space Model): a hybrid deterministic-stochastic latent state that combines the expressiveness of a stochastic latent with the temporal stability of a GRU.

This module implements a simplified version of DreamerV3 with Gaussian latents (instead of categorical) and standard MSE reconstruction loss.

## Architecture

```
RSSM:
  h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})          # deterministic recurrent state
  z_t ~ q(z | h_t, obs_t)                         # posterior (with real obs)
  z_t ~ p(z | h_t)                                # prior (imagination only)

World model heads:
  obs decoder:  (h, z) → obs_hat
  reward head:  (h, z) → r_hat
  continue head: (h, z) → γ_hat

Actor-critic (trained in imagination only):
  imagine H steps: a_t ~ π(h_t, z_t); advance RSSM with prior
  compute λ-returns from imagined rewards + critic bootstrapping
```

## Training Loop

```
for each env step:
    collect real transition → buffer (updates recurrent state with posterior)

    world model update:
        sample sequences → roll RSSM with posterior → KL + recon + reward loss

    actor-critic update:
        warm up latent state → imagine H=15 steps → λ-return → actor/critic loss
```

## Environment

**HalfCheetah-v4** — standard MuJoCo locomotion. Dreamer typically achieves competitive performance using fewer real steps than SAC.

Also supports: **Hopper-v4**, **Walker2d-v4**.

## Key Takeaways

- RSSM combines deterministic (GRU) and stochastic (Gaussian) components: the GRU provides stable temporal context; the Gaussian captures multi-step uncertainty
- KL divergence between posterior and prior is the world model's "surprise" signal
- λ-returns blend Monte Carlo (λ=1) and TD (λ=0) in imagination — λ=0.95 balances bias and variance
- Policy gradients never touch the real env: actor improvement is purely through model backpropagation

## References

- Hafner et al. (2019) — [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
- Hafner et al. (2023) — [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (DreamerV3)
