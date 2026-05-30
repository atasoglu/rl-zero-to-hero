import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm

from agent import HERSACAgent
from rl_common import plot_rewards, watch

ROBOTICS_ENVS = ["FetchReach-v3", "FetchPush-v3", "FetchSlide-v3"]


def make_robotics_env(env_id: str, render_mode=None):
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id, render_mode=render_mode)
    return env


def run_episode(agent: HERSACAgent, env, train: bool = True) -> tuple[float, bool]:
    obs_dict, _ = env.reset()
    total_reward = 0.0
    success = False
    done = False

    while not done:
        obs_flat = agent.obs_to_flat(obs_dict)
        action = agent.select_action(obs_flat, explore=train)
        next_obs_dict, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_obs_flat = agent.obs_to_flat(next_obs_dict)
        next_ag = next_obs_dict["achieved_goal"]

        if train:
            agent.push_step(obs_flat, action, float(reward), next_obs_flat, done, next_ag)
            agent.update()

        obs_dict = next_obs_dict
        total_reward += float(reward)
        if info.get("is_success", False):
            success = True

    if train:
        agent.finalize_episode(env.unwrapped.compute_reward)

    return total_reward, success


def train(agent: HERSACAgent, env, n_episodes: int, render_env=None, render_interval: int = 0) -> tuple[list, list]:
    rewards, successes = [], []
    bar = tqdm(range(n_episodes), desc="Training", unit="ep")
    for ep in bar:
        ep_reward, ep_success = run_episode(agent, env, train=True)
        rewards.append(ep_reward)
        successes.append(float(ep_success))

        if len(rewards) >= 100:
            avg_r = sum(rewards[-100:]) / 100
            avg_s = sum(successes[-100:]) / 100
            bar.set_postfix(reward=f"{ep_reward:.2f}", avg_r=f"{avg_r:.2f}", success=f"{avg_s:.2%}")

        if render_env and render_interval and (ep + 1) % render_interval == 0:
            tqdm.write(f"[episode {ep + 1}] rendering...")
            run_episode(agent, render_env, train=False)

    return rewards, successes


def evaluate(agent: HERSACAgent, env, n_episodes: int = 50) -> tuple[float, float]:
    total_r, total_s = 0.0, 0.0
    for _ in range(n_episodes):
        r, s = run_episode(agent, env, train=False)
        total_r += r
        total_s += float(s)
    return total_r / n_episodes, total_s / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment
    parser.add_argument("--env", default="FetchReach-v3", choices=ROBOTICS_ENVS,
                        help="goal-conditioned robotics environment")

    # Agent hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update rate")
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy temperature (used if --no-auto-alpha)")
    parser.add_argument("--auto-alpha", action=argparse.BooleanOptionalAction, default=True,
                        help="automatic entropy tuning")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=256, help="minibatch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="hidden layer width")
    parser.add_argument("--her-k", type=int, default=4, help="HER: hindsight transitions per real transition")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="compute device")

    # Training
    parser.add_argument("--episodes", type=int, default=200, help="number of training episodes")
    parser.add_argument("--render-interval", type=int, default=0, metavar="N",
                        help="render every N episodes (0=off)")
    parser.add_argument("--watch", type=int, default=5, metavar="N",
                        help="watch trained agent for N episodes after training")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    env = make_robotics_env(args.env)
    obs_size = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = HERSACAgent(
        obs_size=obs_size,
        goal_dim=goal_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        auto_tune_alpha=args.auto_alpha,
        alpha=args.alpha,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        her_k=args.her_k,
        device=device,
    )

    render_env = make_robotics_env(args.env, render_mode="human") if args.render_interval else None
    rewards, successes = train(agent, env, args.episodes, render_env=render_env, render_interval=args.render_interval)
    if render_env:
        render_env.close()

    eval_env = make_robotics_env(args.env)
    mean_reward, mean_success = evaluate(agent, eval_env, n_episodes=50)
    print(f"{args.env} | device: {device} | episodes: {args.episodes} | "
          f"eval reward: {mean_reward:.2f} | success rate: {mean_success:.2%}")

    plot_rewards(rewards, title=f"HER+SAC on {args.env}")

    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_robotics_env(args.env, render_mode="human")
        watch(lambda env: run_episode(agent, env, train=False)[0], render_env, n_episodes=args.watch)
