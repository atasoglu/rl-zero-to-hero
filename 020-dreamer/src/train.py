import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from tqdm import tqdm

from agent import DreamerAgent
from rl_common import make_env, plot_rewards, watch

MUJOCO_ENVS = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]


def run_episode(agent: DreamerAgent, env, train: bool = True) -> float:
    obs, _ = env.reset()
    agent.reset_state()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(obs, explore=train)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if train:
            agent.push(obs, action, float(reward), next_obs, done)
            agent.update()
        obs = next_obs
        total_reward += float(reward)
    return total_reward


def train(agent: DreamerAgent, env, n_episodes: int,
          render_env=None, render_interval: int = 0) -> list[float]:
    rewards = []
    bar = tqdm(range(n_episodes), desc="Training", unit="ep")
    for ep in bar:
        ep_reward = run_episode(agent, env, train=True)
        rewards.append(ep_reward)
        if len(rewards) >= 10:
            avg = sum(rewards[-10:]) / 10
            bar.set_postfix(reward=f"{ep_reward:.1f}", avg10=f"{avg:.1f}")
        if render_env and render_interval and (ep + 1) % render_interval == 0:
            tqdm.write(f"[episode {ep + 1}] rendering...")
            run_episode(agent, render_env, train=False)
    return rewards


def evaluate(agent: DreamerAgent, env, n_episodes: int = 10) -> float:
    return sum(run_episode(agent, env, train=False) for _ in range(n_episodes)) / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--env", default="HalfCheetah-v4", choices=MUJOCO_ENVS)
    parser.add_argument("--h-dim", type=int, default=200, help="RSSM deterministic state dim")
    parser.add_argument("--z-dim", type=int, default=30, help="RSSM stochastic state dim")
    parser.add_argument("--lr-model", type=float, default=6e-4)
    parser.add_argument("--lr-actor", type=float, default=8e-5)
    parser.add_argument("--lr-critic", type=float, default=8e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--imagine-horizon", type=int, default=15,
                        help="imagination rollout length for actor-critic")
    parser.add_argument("--seq-len", type=int, default=50,
                        help="sequence length for world model training")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--render-interval", type=int, default=0, metavar="N")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--watch", type=int, default=5, metavar="N")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    env = make_env(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = DreamerAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        lr_model=args.lr_model,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        imagine_horizon=args.imagine_horizon,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=device,
    )

    render_env = make_env(args.env, render_mode="human") if args.render_interval else None
    rewards = train(agent, env, args.episodes, render_env=render_env, render_interval=args.render_interval)
    if render_env:
        render_env.close()

    eval_env = make_env(args.env)
    mean_reward = evaluate(agent, eval_env)
    print(f"{args.env} | device: {device} | episodes: {args.episodes} | eval mean reward: {mean_reward:.1f}")

    plot_rewards(rewards, title=f"Dreamer on {args.env}")
    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(args.env, render_mode="human")
        watch(lambda env: run_episode(agent, env, train=False), render_env, n_episodes=args.watch)
