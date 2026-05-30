# 018 — MBPO (Model-Based Policy Optimization)

> Use an ensemble of learned dynamics models to generate short synthetic rollouts, augmenting real data for SAC training.

## Overview

MBPO (Janner et al., 2019) achieves the sample efficiency of model-based RL while retaining the asymptotic performance of model-free RL. The key idea: instead of long model rollouts (which accumulate error), generate many **short rollouts** (horizon H=1–5) starting from real states, and mix the synthetic data with real data for policy training.

An **ensemble** of 5 probabilistic dynamics models is used. Each member predicts `(Δobs, reward)` as a Gaussian; using multiple members provides epistemic uncertainty estimates that implicitly limit rollout reliability.

## Algorithm

```
Initialize SAC policy, ensemble dynamics model, real buffer, model buffer

for each env step:
    collect real transition → real buffer
    if step % model_train_freq == 0:
        train ensemble on real buffer (50 gradient steps)
        for rollout_batch starting states from real buffer:
            roll out H steps with current policy through ensemble
            store synthetic transitions → model buffer
    
    sample batch: 80% from model buffer + 20% from real buffer
    SAC update
```

## Architecture

**EnsembleDynamicsModel**: 5 independent MLPs, each outputting `(mean, log_var)` for `(Δobs ‖ reward)`. Uses SiLU activations and learnable log_var bounds for numerical stability.

## Environment

**Hopper-v4** — standard MuJoCo locomotion benchmark. MBPO typically matches SAC performance with ~5× fewer real environment steps.

Also supports: **HalfCheetah-v4**, **Walker2d-v4**.

## Key Takeaways

- Short rollouts (H≈1) prevent model error from compounding while still providing useful synthetic data
- Ensemble disagreement provides implicit uncertainty — high-variance predictions = unreliable regions
- Mixing real (20%) + model (80%) data keeps the policy anchored to real transitions
- MBPO is the canonical "model-based achieves model-free performance" result on MuJoCo benchmarks

## References

- Janner et al. (2019) — [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
