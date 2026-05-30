import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from agent import QLearningAgent
from rl_common import make_env, plot_rewards, watch


def run_episode(agent: QLearningAgent, env, train: bool = True) -> float:
    state, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if train:
            agent.update(state, action, float(reward), next_state, done)
        state = next_state
        total_reward += float(reward)
    if train:
        agent.decay_epsilon()
    return total_reward


def train(agent: QLearningAgent, env, n_episodes: int, render_env=None, render_interval: int = 0) -> list[float]:
    rewards = []
    bar = tqdm(range(n_episodes), desc="Training", unit="ep")
    for ep in bar:
        ep_reward = run_episode(agent, env, train=True)
        rewards.append(ep_reward)

        if len(rewards) >= 100:
            avg = sum(rewards[-100:]) / 100
            bar.set_postfix(reward=f"{ep_reward:.2f}", avg100=f"{avg:.2f}", eps=f"{agent.epsilon:.3f}")

        if render_env and render_interval and (ep + 1) % render_interval == 0:
            saved_eps = agent.epsilon
            agent.epsilon = 0.05
            tqdm.write(f"[episode {ep + 1}] rendering... (ε={agent.epsilon:.2f})")
            run_episode(agent, render_env, train=False)
            agent.epsilon = saved_eps

    return rewards


def evaluate(agent: QLearningAgent, env, n_episodes: int = 100) -> float:
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    total = sum(run_episode(agent, env, train=False) for _ in range(n_episodes))
    agent.epsilon = saved_eps
    return total / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment
    parser.add_argument("--env", default="FrozenLake-v1", choices=["FrozenLake-v1", "Taxi-v3"], help="gym environment")
    parser.add_argument("--slippery", action=argparse.BooleanOptionalAction, default=True, help="FrozenLake: stochastic transitions (--no-slippery for deterministic)")
    parser.add_argument("--map", default="4x4", choices=["4x4", "8x8"], dest="map_name", help="FrozenLake: map size")

    # Agent hyperparameters
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="epsilon multiplier per episode")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="minimum epsilon")

    # Training
    parser.add_argument("--episodes", type=int, default=5000, help="number of training episodes")
    parser.add_argument("--render-interval", type=int, default=0, metavar="N", help="render every N episodes (0=off)")
    parser.add_argument("--watch", type=int, default=0, metavar="N", help="watch trained agent for N episodes after training")
    args = parser.parse_args()

    env_kwargs = {}
    if args.env == "FrozenLake-v1":
        env_kwargs["is_slippery"] = args.slippery
        env_kwargs["map_name"] = args.map_name

    env = make_env(args.env, **env_kwargs)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    render_env = make_env(args.env, render_mode="human", **env_kwargs) if args.render_interval else None
    rewards = train(agent, env, args.episodes, render_env=render_env, render_interval=args.render_interval)
    if render_env:
        render_env.close()

    eval_env = make_env(args.env, **env_kwargs)
    mean_reward = evaluate(agent, eval_env, n_episodes=100)
    print(f"{args.env} | episodes: {args.episodes} | eval mean reward: {mean_reward:.3f}")

    plot_rewards(rewards, title=f"Q-Learning on {args.env}")

    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(args.env, render_mode="human", **env_kwargs)
        watch(lambda env: run_episode(agent, env, train=False), render_env, n_episodes=args.watch)
