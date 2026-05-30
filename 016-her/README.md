# 016 — Hindsight Experience Replay (HER)

> Relabel failed episodes with the goals that were actually achieved, enabling learning from sparse rewards without reward shaping.

## Overview

Goal-conditioned tasks with sparse rewards (e.g. "reach position X") are notoriously hard: the agent rarely receives a positive reward signal early in training. HER (Andrychowicz et al., 2017) solves this elegantly:

After a failed episode where the agent reached position Y instead of goal X, HER asks: *"What if Y had been the goal?"* It adds relabeled transitions to the replay buffer where the desired goal is replaced by the achieved goal, generating useful learning signal even from failures.

This module uses **SAC + HER** on robotic manipulation tasks from `gymnasium-robotics`.

## Algorithm

```
for each episode:
    run policy π(obs ‖ goal) until done
    store all transitions in episode buffer

    # Original transitions (with real goal)
    push all (obs, a, r, obs', done) to replay buffer

    # Hindsight transitions (future strategy)
    for each step t:
        pick k random future steps j ≥ t
        new_goal = achieved_goal[j]
        new_reward = compute_reward(achieved_goal[t+1], new_goal)
        push relabeled (obs[t] ‖ new_goal, a, new_r, obs'[t] ‖ new_goal, done)

    update SAC policy from replay buffer
```

## Environment

**FetchReach-v3** — 7-DoF arm, 3D target position, sparse binary reward (0 if reached, −1 otherwise). The observation is a dict with `"observation"`, `"achieved_goal"`, `"desired_goal"`.

Also supports: **FetchPush-v3**, **FetchSlide-v3**.

## Key Takeaways

- [ ] HER converts any goal-conditioned environment into a dense learning problem without modifying the reward function
- [ ] The "future" strategy (goal = achieved_goal at a random future step) works best in practice
- [ ] Success rate is a better metric than mean reward for sparse-reward tasks
- [ ] HER is compatible with any off-policy algorithm (SAC, DDPG, TD3); SAC is used here for its stability

## References

- Andrychowicz et al. (2017) — [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
