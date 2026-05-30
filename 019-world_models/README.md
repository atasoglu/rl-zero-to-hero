# 019 — World Models (Ha & Schmidhuber 2018)

> Learn a compressed visual world model (VAE + MDN-RNN), then train a controller entirely within the model's imagination.

## Overview

Ha & Schmidhuber (2018) decompose the agent into three components:

- **V (VAE)**: Compresses each raw frame into a small latent vector `z`.
- **M (MDN-RNN)**: Predicts the next latent `z_{t+1}` given `(z_t, a_t, h_t)` using a Mixture Density Network over a GRU hidden state.
- **C (Controller)**: A simple linear layer mapping `(z, h)` → actions, trained with PPO inside the imagination.

The controller is trained entirely in **latent space** — no further environment interaction needed after world model training.

## Training Stages

```
Stage 1: Collect random rollouts from real environment
Stage 2: Train VAE to encode/decode frames (reconstruction + KL loss)
Stage 3: Train MDN-RNN on sequences of (z, a) pairs
Stage 4: Train PPO controller using (z, h) from real env rollouts
```

Use `--stage` to run stages independently (useful for iterative improvement).

## Architecture

| Component | Input | Output |
|-----------|-------|--------|
| VAE Encoder | (3, 64, 64) frame | z ∈ ℝ³² |
| VAE Decoder | z ∈ ℝ³² | (3, 64, 64) reconstruction |
| MDN-RNN | (z_t, a_t, h_t) | GMM over z_{t+1}, r̂, ĥ |
| Controller | (z, h) | action logits + value |

## Environment

**CarRacing-v3** (discrete actions) — top-down racing with pixel observations. The VAE captures road geometry; the RNN predicts future track layout.

## Key Takeaways

- [ ] Separating perception (V), memory (M), and control (C) enables modular world model training
- [ ] The MDN-RNN's mixture distribution captures multimodal futures (e.g. branching roads)
- [ ] Dream training: controllers trained in imagination can transfer to the real environment
- [ ] The linear controller is intentionally simple — the world model does the heavy lifting

## References

- Ha & Schmidhuber (2018) — [World Models](https://arxiv.org/abs/1803.10122)
