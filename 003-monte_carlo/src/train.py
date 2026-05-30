import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from agent import MonteCarloAgent
from rl_common import make_env, plot_rewards, watch

ENV_ID = "Blackjack-v1"


def run_episode(agent: MonteCarloAgent, env, train: bool = True) -> float:
    state, _ = env.reset()
    episode = []
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if train:
            episode.append((state, action, float(reward)))
        state = next_state
        total_reward += float(reward)
    if train:
        agent.update(episode)
        agent.decay_epsilon()
    return total_reward


def train(agent: MonteCarloAgent, env, n_episodes: int, render_env=None, render_interval: int = 0) -> list[float]:
    rewards = []
    bar = tqdm(range(n_episodes), desc="Training", unit="ep")
    for ep in bar:
        ep_reward = run_episode(agent, env, train=True)
        rewards.append(ep_reward)

        if len(rewards) >= 1000:
            avg = sum(rewards[-1000:]) / 1000
            bar.set_postfix(reward=f"{ep_reward:.2f}", avg1k=f"{avg:.3f}", eps=f"{agent.epsilon:.4f}")

        if render_env and render_interval and (ep + 1) % render_interval == 0:
            saved_eps = agent.epsilon
            agent.epsilon = 0.05
            tqdm.write(f"[episode {ep + 1}] rendering... (ε={agent.epsilon:.2f})")
            run_episode(agent, render_env, train=False)
            agent.epsilon = saved_eps

    return rewards


def evaluate(agent: MonteCarloAgent, env, n_episodes: int = 1000) -> float:
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    total = sum(run_episode(agent, env, train=False) for _ in range(n_episodes))
    agent.epsilon = saved_eps
    return total / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment
    parser.add_argument("--natural", action=argparse.BooleanOptionalAction, default=False, help="Blackjack: bonus reward for a natural (21 on first two cards)")
    parser.add_argument("--sab", action=argparse.BooleanOptionalAction, default=False, help="Blackjack: follow Sutton & Barto rules (dealer sticks on soft 17)")

    # Agent hyperparameters
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor (1.0 = undiscounted, standard for Blackjack)")
    parser.add_argument("--epsilon-decay", type=float, default=0.9999, help="epsilon multiplier per episode")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="minimum epsilon")

    # Training
    parser.add_argument("--episodes", type=int, default=500000, help="number of training episodes")
    parser.add_argument("--render-interval", type=int, default=0, metavar="N", help="render every N episodes (0=off)")
    parser.add_argument("--watch", type=int, default=0, metavar="N", help="watch trained agent for N episodes after training")
    args = parser.parse_args()

    env_kwargs = {"natural": args.natural, "sab": args.sab}

    env = make_env(ENV_ID, **env_kwargs)
    agent = MonteCarloAgent(
        n_actions=env.action_space.n,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    render_env = make_env(ENV_ID, render_mode="human", **env_kwargs) if args.render_interval else None
    rewards = train(agent, env, args.episodes, render_env=render_env, render_interval=args.render_interval)
    if render_env:
        render_env.close()

    eval_env = make_env(ENV_ID, **env_kwargs)
    mean_reward = evaluate(agent, eval_env, n_episodes=10000)
    print(f"{ENV_ID} | episodes: {args.episodes} | eval mean reward: {mean_reward:.4f}")

    plot_rewards(rewards, title=f"Monte Carlo Control on {ENV_ID}", window=500)

    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(ENV_ID, render_mode="human", **env_kwargs)
        watch(lambda env: run_episode(agent, env, train=False), render_env, n_episodes=args.watch)
