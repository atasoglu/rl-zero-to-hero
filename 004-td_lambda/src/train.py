import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from agent import TDLambdaAgent
from env import discretize, make_env, n_states
from rl_common import plot_rewards, watch

ENV_ID = "CartPole-v1"


def run_episode(agent: TDLambdaAgent, env, train: bool = True) -> float:
    obs, _ = env.reset()
    state = discretize(obs)
    action = agent.select_action(state)
    if train:
        agent.reset_traces()
    total_reward = 0.0
    done = False
    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)
        next_action = agent.select_action(next_state)
        if train:
            agent.update(state, action, float(reward), next_state, next_action, done)
        state = next_state
        action = next_action
        total_reward += float(reward)
    if train:
        agent.decay_epsilon()
    return total_reward


def train(agent: TDLambdaAgent, env, n_episodes: int, render_env=None, render_interval: int = 0) -> list[float]:
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


def evaluate(agent: TDLambdaAgent, env, n_episodes: int = 100) -> float:
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    total = sum(run_episode(agent, env, train=False) for _ in range(n_episodes))
    agent.epsilon = saved_eps
    return total / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment — CartPole has no configurable env params exposed by gym
    # but discretization bin count affects the state space size
    parser.add_argument("--n-bins", type=int, default=10, help="CartPole: number of bins per observation dimension for discretization")

    # Agent hyperparameters
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lam", type=float, default=0.9, help="eligibility trace decay (0=TD(0), 1≈MC)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="epsilon multiplier per episode")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="minimum epsilon")

    # Training
    parser.add_argument("--episodes", type=int, default=3000, help="number of training episodes")
    parser.add_argument("--render-interval", type=int, default=0, metavar="N", help="render every N episodes (0=off)")
    parser.add_argument("--watch", type=int, default=5, metavar="N", help="watch trained agent for N episodes after training")
    args = parser.parse_args()

    env = make_env(ENV_ID)
    agent = TDLambdaAgent(
        n_states=n_states(args.n_bins),
        n_actions=env.action_space.n,
        alpha=args.alpha,
        gamma=args.gamma,
        lam=args.lam,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    render_env = make_env(ENV_ID, render_mode="human") if args.render_interval else None
    rewards = train(agent, env, args.episodes, render_env=render_env, render_interval=args.render_interval)
    if render_env:
        render_env.close()

    eval_env = make_env(ENV_ID)
    mean_reward = evaluate(agent, eval_env, n_episodes=100)
    print(f"{ENV_ID} | λ={args.lam} | episodes: {args.episodes} | eval mean reward: {mean_reward:.1f}")

    plot_rewards(rewards, title=f"TD(λ) on {ENV_ID}")

    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(ENV_ID, render_mode="human")
        watch(lambda env: run_episode(agent, env, train=False), render_env, n_episodes=args.watch)
