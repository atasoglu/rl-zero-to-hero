import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from tqdm import tqdm

from agent import DoubleDQNAgent
from rl_common import make_env, plot_rewards, watch


def run_episode(agent: DoubleDQNAgent, env, train: bool = True) -> float:
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if train:
            agent.push(obs, action, reward, next_obs, done)
            agent.update()
        obs = next_obs
        total_reward += float(reward)
    if train:
        agent.decay_epsilon()
    return total_reward


def train(agent: DoubleDQNAgent, env, n_episodes: int, render_env=None, render_interval: int = 0) -> list[float]:
    rewards = []
    bar = tqdm(range(n_episodes), desc="Training", unit="ep")
    for ep in bar:
        ep_reward = run_episode(agent, env, train=True)
        rewards.append(ep_reward)

        if len(rewards) >= 100:
            avg = sum(rewards[-100:]) / 100
            bar.set_postfix(reward=f"{ep_reward:.1f}", avg100=f"{avg:.1f}", eps=f"{agent.epsilon:.3f}")

        if render_env and render_interval and (ep + 1) % render_interval == 0:
            saved_eps = agent.epsilon
            agent.epsilon = 0.05
            tqdm.write(f"[episode {ep + 1}] rendering... (ε={agent.epsilon:.2f})")
            run_episode(agent, render_env, train=False)
            agent.epsilon = saved_eps

    return rewards


def evaluate(agent: DoubleDQNAgent, env, n_episodes: int = 10) -> float:
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    total = sum(run_episode(agent, env, train=False) for _ in range(n_episodes))
    agent.epsilon = saved_eps
    return total / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment
    parser.add_argument("--env", default="LunarLander-v3", choices=["CartPole-v1", "LunarLander-v3"],
                        help="gym environment")

    # Agent hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--epsilon-decay", type=float, default=0.998, help="epsilon multiplier per episode")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="minimum epsilon")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=64, help="minibatch size for updates")
    parser.add_argument("--target-update-freq", type=int, default=100, help="copy online net to target every N updates")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden layer width")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="compute device (auto = cuda > mps > cpu)")

    # Training
    parser.add_argument("--episodes", type=int, default=600, help="number of training episodes")
    parser.add_argument("--render-interval", type=int, default=0, metavar="N",
                        help="render every N episodes (0=off)")
    parser.add_argument("--watch", type=int, default=5, metavar="N",
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

    agent = DoubleDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
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

    plot_rewards(rewards, title=f"Double DQN on {args.env}")

    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(args.env, render_mode="human")
        watch(lambda env: run_episode(agent, env, train=False), render_env, n_episodes=args.watch)
