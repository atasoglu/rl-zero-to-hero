import gymnasium as gym


def make_env(env_id: str, render_mode: str | None = None, seed: int = 42, **kwargs) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env
