import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm

from agent import ICMAgent
from rl_common import plot_rewards

ATARI_ENVS = ["ALE/MontezumaRevenge-v5", "ALE/Pitfall-v5", "ALE/PrivateEye-v5"]


def make_atari_env(env_id: str, render_mode=None):
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make(env_id, render_mode=render_mode, obs_type="grayscale",
                   frameskip=4, repeat_action_probability=0.0)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=42)
    return env


def collect_rollout(agent: ICMAgent, env, n_steps: int) -> tuple[dict, float, int]:
    obs_buf, next_obs_buf = [], []
    actions_buf, log_probs_buf, values_buf = [], [], []
    ext_rewards_buf, dones_buf = [], []

    obs, _ = env.reset()
    total_ext_reward = 0.0
    n_episodes = 0

    for _ in range(n_steps):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_buf.append(obs)
        next_obs_buf.append(next_obs)
        actions_buf.append(action)
        log_probs_buf.append(log_prob)
        values_buf.append(value)
        ext_rewards_buf.append(float(reward))
        dones_buf.append(float(done))
        total_ext_reward += float(reward)

        if done:
            obs, _ = env.reset()
            n_episodes += 1
        else:
            obs = next_obs

    obs_arr = np.array(obs_buf)
    next_obs_arr = np.array(next_obs_buf)
    actions_arr = np.array(actions_buf)

    r_int = agent.compute_intrinsic_rewards(obs_arr, actions_arr, next_obs_arr)
    total_rewards = np.array(ext_rewards_buf) + r_int

    return {
        "obs": obs_arr,
        "next_obs": next_obs_arr,
        "actions": actions_arr,
        "log_probs": np.array(log_probs_buf),
        "values": np.array(values_buf),
        "total_rewards": total_rewards,
        "dones": np.array(dones_buf),
    }, total_ext_reward, max(n_episodes, 1)


def train(agent: ICMAgent, env, n_updates: int, n_steps: int) -> list[float]:
    rewards_per_update = []
    bar = tqdm(range(n_updates), desc="Training", unit="update")
    for upd in bar:
        rollout, total_ext, n_eps = collect_rollout(agent, env, n_steps)
        agent.update(rollout)
        avg_ext = total_ext / n_eps
        rewards_per_update.append(avg_ext)
        if len(rewards_per_update) >= 10:
            avg10 = sum(rewards_per_update[-10:]) / 10
            bar.set_postfix(ext_reward=f"{avg_ext:.1f}", avg10=f"{avg10:.1f}")
    return rewards_per_update


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--env", default="ALE/MontezumaRevenge-v5", choices=ATARI_ENVS,
                        help="hard-exploration Atari environment")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--n-steps", type=int, default=128, help="steps per rollout")
    parser.add_argument("--n-updates", type=int, default=1000, help="number of PPO updates")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=256, help="minibatch size")
    parser.add_argument("--clip-eps", type=float, default=0.1, help="PPO clip epsilon")
    parser.add_argument("--icm-eta", type=float, default=0.01, help="intrinsic reward scale")
    parser.add_argument("--icm-lambda", type=float, default=0.1, help="ICM loss weight")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--watch", type=int, default=3, metavar="N",
                        help="watch trained agent for N episodes (0=skip)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    env = make_atari_env(args.env)
    n_actions = env.action_space.n

    agent = ICMAgent(
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_eps=args.clip_eps,
        icm_eta=args.icm_eta,
        icm_lambda=args.icm_lambda,
        device=device,
    )

    rewards = train(agent, env, args.n_updates, args.n_steps)
    env.close()

    print(f"{args.env} | device: {device} | updates: {args.n_updates} | "
          f"final avg ext reward: {sum(rewards[-10:]) / min(10, len(rewards)):.1f}")

    plot_rewards(rewards, title=f"ICM+PPO on {args.env}")

    if args.watch > 0:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        render_env = make_atari_env(args.env, render_mode="human")
        for ep in range(args.watch):
            obs, _ = render_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _, _ = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = render_env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)
            print(f"  watch episode {ep + 1}: reward={ep_reward:.1f}")
        render_env.close()
