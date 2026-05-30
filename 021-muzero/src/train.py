import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm

from agent import MuZeroAgent
from rl_common import make_env, plot_rewards, watch


def run_episode(agent: MuZeroAgent, env, train: bool = True,
                temperature: float = 1.0) -> float:
    obs, _ = env.reset()
    trajectory = []
    total_reward = 0.0
    done = False

    while not done:
        action, action_probs = agent.select_action(obs, temperature=temperature if train else 0.0)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if train:
            trajectory.append({
                "obs": obs,
                "action": action,
                "action_probs": action_probs,
                "reward": float(reward),
            })

        obs = next_obs
        total_reward += float(reward)

    if train:
        agent.store(trajectory)
        agent.update()

    return total_reward


def train(agent: MuZeroAgent, env, n_episodes: int) -> list[float]:
    rewards = []
    # Anneal temperature: start 1.0, decay to 0.1
    bar = tqdm(range(n_episodes), desc="Training", unit="ep")
    for ep in bar:
        temp = max(0.1, 1.0 - ep / n_episodes)
        ep_reward = run_episode(agent, env, train=True, temperature=temp)
        rewards.append(ep_reward)
        if len(rewards) >= 100:
            avg = sum(rewards[-100:]) / 100
            bar.set_postfix(reward=f"{ep_reward:.1f}", avg100=f"{avg:.1f}", temp=f"{temp:.2f}")
    return rewards


def evaluate(agent: MuZeroAgent, env, n_episodes: int = 10) -> float:
    return sum(run_episode(agent, env, train=False) for _ in range(n_episodes)) / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--env", default="CartPole-v1",
                        choices=["CartPole-v1", "LunarLander-v3", "Acrobot-v1"])
    parser.add_argument("--state-dim", type=int, default=64, help="hidden state dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="network hidden width")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.997, help="discount factor")
    parser.add_argument("--simulations", type=int, default=50,
                        help="MCTS simulations per action")
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--td-steps", type=int, default=5, help="n-step return depth")
    parser.add_argument("--episodes", type=int, default=300, help="training episodes")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--watch", type=int, default=5, metavar="N")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    env = make_env(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = MuZeroAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        n_simulations=args.simulations,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        td_steps=args.td_steps,
        device=device,
    )

    rewards = train(agent, env, args.episodes)
    env.close()

    eval_env = make_env(args.env)
    mean_reward = evaluate(agent, eval_env)
    print(f"{args.env} | device: {device} | episodes: {args.episodes} | "
          f"eval mean reward: {mean_reward:.1f}")

    plot_rewards(rewards, title=f"MuZero on {args.env}")
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(args.env, render_mode="human")
        watch(lambda env: run_episode(agent, env, train=False), render_env, n_episodes=args.watch)
