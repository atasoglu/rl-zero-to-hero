# 011 — Trust Region Policy Optimization (TRPO)

> Guarantee monotonic policy improvement by constraining each update to stay within a trust region defined by a KL divergence bound.

## Overview

A2C and REINFORCE take gradient steps without constraining how much the policy changes. A large step can overshoot and collapse the policy — the agent suddenly performs much worse, and recovery is slow. TRPO (Schulman et al., 2015) formalizes this problem: it finds the largest policy improvement step that keeps the new policy within a KL divergence bound of the old one.

The optimization problem is:

```
maximize  E[ (π_new / π_old) * A(s, a) ]
subject to  E[ KL(π_old || π_new) ] <= δ
```

Solving this exactly requires second-order methods. TRPO uses the **conjugate gradient** algorithm to compute the natural gradient direction and a **line search** to satisfy the KL constraint.

## Algorithm

```
for each update iteration (every k episodes):
    collect trajectories with current policy
    compute advantages with GAE
    compute policy gradient g and Fisher-vector product Fvp

    # Natural gradient via conjugate gradient
    step_dir = CG(Fvp, -g)
    step_size = sqrt(2 * delta / (step_dir^T * Fvp(step_dir)))
    step = step_dir * step_size

    # Backtracking line search to satisfy KL constraint
    while KL(π_old || π_new) > delta or surrogate loss increases:
        step_size *= 0.5
        update policy

    # Update value function with several gradient steps
    fit V(s) to observed returns
```

## Environment

**CartPole-v1** — fast sanity check to verify the constrained update mechanism.

**LunarLander-v3** — better demonstrates TRPO's stability advantage.

**MuJoCo** (HalfCheetah-v4, Hopper-v4, Walker2d-v4) — the original benchmark. Note: this implementation uses discrete actions; for continuous action TRPO see standard TRPO with Gaussian policies.

### MuJoCo setup

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install gymnasium[mujoco]
```

## Key Takeaways

- [ ] TRPO guarantees monotonic improvement: each update cannot make the policy worse (in expectation under the constraint)
- [ ] The KL constraint replaces the learning rate — `max_kl` is the key hyperparameter
- [ ] Conjugate gradient solves the trust-region subproblem without forming the full Fisher matrix (O(n) Hessian-vector products vs O(n²) storage)
- [ ] The Fisher information matrix defines the geometry of policy space; natural gradient steps are invariant to parameterization
- [ ] TRPO is complex to implement correctly; PPO (module 012) achieves similar stability with a much simpler first-order method
- [ ] GAE (Generalized Advantage Estimation) reduces variance of the advantage estimate by trading in some bias

## References

- Schulman et al. (2015) — [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- Schulman et al. (2015) — [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (GAE)
