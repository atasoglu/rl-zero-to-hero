import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from tqdm import tqdm

from agent import REINFORCEAgent
from rl_common import make_env, plot_rewards, watch


def run_episode(agent: REINFORCEAgent, env, train: bool = True) -> float:
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if train:
            agent.store_reward(float(reward))
        obs = next_obs
        total_reward += float(reward)
    if train:
        agent.update()
    return total_reward


def train(agent: REINFORCEAgent, env, n_episodes: int, render_env=None, render_interval: int = 0) -> list[float]:
    rewards = []
    bar = tqdm(range(n_episodes), desc="Training", unit="ep")
    for ep in bar:
        ep_reward = run_episode(agent, env, train=True)
        rewards.append(ep_reward)

        if len(rewards) >= 100:
            avg = sum(rewards[-100:]) / 100
            bar.set_postfix(reward=f"{ep_reward:.1f}", avg100=f"{avg:.1f}")

        if render_env and render_interval and (ep + 1) % render_interval == 0:
            tqdm.write(f"[episode {ep + 1}] rendering...")
            run_episode(agent, render_env, train=False)

    return rewards


def evaluate(agent: REINFORCEAgent, env, n_episodes: int = 10) -> float:
    return sum(run_episode(agent, env, train=False) for _ in range(n_episodes)) / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment
    parser.add_argument("--env", default="CartPole-v1", choices=["CartPole-v1", "LunarLander-v3"],
                        help="gym environment")

    # Agent hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden layer width")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="compute device (auto = cuda > mps > cpu)")

    # Training
    parser.add_argument("--episodes", type=int, default=1000, help="number of training episodes")
    parser.add_argument("--render-interval", type=int, default=0, metavar="N",
                        help="render every N episodes (0=off)")
    parser.add_argument("--watch", type=int, default=0, metavar="N",
                        help="watch trained agent for N episodes after training")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    env = make_env(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = REINFORCEAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    render_env = make_env(args.env, render_mode="human") if args.render_interval else None
    rewards = train(agent, env, args.episodes, render_env=render_env, render_interval=args.render_interval)
    if render_env:
        render_env.close()

    eval_env = make_env(args.env)
    mean_reward = evaluate(agent, eval_env, n_episodes=10)
    print(f"{args.env} | device: {device} | episodes: {args.episodes} | eval mean reward: {mean_reward:.1f}")

    plot_rewards(rewards, title=f"REINFORCE on {args.env}")

    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(args.env, render_mode="human")
        watch(lambda env: run_episode(agent, env, train=False), render_env, n_episodes=args.watch)
