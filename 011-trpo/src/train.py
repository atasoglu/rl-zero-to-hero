import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm

from agent import TRPOAgent
from rl_common import make_env, plot_rewards, watch

MUJOCO_ENVS = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]


def collect_episode(agent: TRPOAgent, env) -> tuple[dict, float]:
    obs_list, action_list, log_prob_list, reward_list, done_list = [], [], [], [], []
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, log_prob = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs_list.append(obs)
        action_list.append(action)
        log_prob_list.append(log_prob)
        reward_list.append(float(reward))
        done_list.append(float(done))
        obs = next_obs
        total_reward += float(reward)
    return {
        "obs": np.array(obs_list),
        "actions": np.array(action_list),
        "log_probs": np.array(log_prob_list),
        "rewards": np.array(reward_list),
        "dones": np.array(done_list),
    }, total_reward


def run_episode_eval(agent: TRPOAgent, env) -> float:
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
    return total_reward


def train(
    agent: TRPOAgent, env, n_episodes: int, episodes_per_update: int = 5,
    render_env=None, render_interval: int = 0
) -> list[float]:
    rewards = []
    bar = tqdm(range(n_episodes), desc="Training", unit="ep")
    batch = []
    for ep in bar:
        traj, ep_reward = collect_episode(agent, env)
        batch.append(traj)
        rewards.append(ep_reward)

        if len(batch) >= episodes_per_update:
            agent.update(batch)
            batch = []

        if len(rewards) >= 100:
            avg = sum(rewards[-100:]) / 100
            bar.set_postfix(reward=f"{ep_reward:.1f}", avg100=f"{avg:.1f}")

        if render_env and render_interval and (ep + 1) % render_interval == 0:
            tqdm.write(f"[episode {ep + 1}] rendering...")
            run_episode_eval(agent, render_env)

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment
    parser.add_argument("--env", default="CartPole-v1",
                        choices=["CartPole-v1", "LunarLander-v3"] + MUJOCO_ENVS,
                        help="gym environment (MuJoCo envs require gymnasium[mujoco])")

    # Agent hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.97, help="GAE lambda")
    parser.add_argument("--max-kl", type=float, default=0.01, help="maximum KL divergence per update")
    parser.add_argument("--damping", type=float, default=0.1, help="damping for Fisher-vector product")
    parser.add_argument("--value-lr", type=float, default=1e-3, help="value network learning rate")
    parser.add_argument("--value-iters", type=int, default=5, help="value network gradient steps per update")
    parser.add_argument("--hidden-dim", type=int, default=64, help="hidden layer width")
    parser.add_argument("--episodes-per-update", type=int, default=5, help="episodes to collect before each policy update")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="compute device (auto = cuda > mps > cpu)")

    # Training
    parser.add_argument("--episodes", type=int, default=500, help="number of training episodes")
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

    agent = TRPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_kl=args.max_kl,
        damping=args.damping,
        value_lr=args.value_lr,
        value_iters=args.value_iters,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    render_env = make_env(args.env, render_mode="human") if args.render_interval else None
    rewards = train(
        agent, env, args.episodes, episodes_per_update=args.episodes_per_update,
        render_env=render_env, render_interval=args.render_interval
    )
    if render_env:
        render_env.close()

    eval_env = make_env(args.env)
    mean_reward = sum(run_episode_eval(agent, eval_env) for _ in range(10)) / 10
    print(f"{args.env} | device: {device} | episodes: {args.episodes} | eval mean reward: {mean_reward:.1f}")

    plot_rewards(rewards, title=f"TRPO on {args.env}")

    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(args.env, render_mode="human")
        watch(lambda env: run_episode_eval(agent, env), render_env, n_episodes=args.watch)
