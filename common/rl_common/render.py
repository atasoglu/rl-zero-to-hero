import time
from typing import Callable

import gymnasium as gym


def watch(run_fn: Callable[[gym.Env], float], env: gym.Env, n_episodes: int = 5):
    """
    Watch a trained agent.

    run_fn: closure over the agent — takes env, runs one episode, returns total reward
    env:    pre-created with render_mode="human"
    """
    print(f"\nWatching trained agent for {n_episodes} episode(s)...")
    for ep in range(n_episodes):
        reward = run_fn(env)
        print(f"  episode {ep + 1}: reward = {reward:.1f}")
        time.sleep(0.5)
    env.close()
