import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from tqdm import tqdm

from agent import MBPOAgent
from rl_common import make_env, plot_rewards, watch

MUJOCO_ENVS = ["Hopper-v4", "HalfCheetah-v4", "Walker2d-v4"]


def run_episode(agent: MBPOAgent, env, train: bool = True) -> float:
    obs, _ = env.reset()
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


def train(agent: MBPOAgent, env, n_episodes: int, render_env=None, render_interval: int = 0) -> list[float]:
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


def evaluate(agent: MBPOAgent, env, n_episodes: int = 10) -> float:
    return sum(run_episode(agent, env, train=False) for _ in range(n_episodes)) / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--env", default="Hopper-v4", choices=MUJOCO_ENVS)
    parser.add_argument("--lr", type=float, default=3e-4, help="SAC learning rate")
    parser.add_argument("--model-lr", type=float, default=1e-3, help="dynamics model learning rate")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-members", type=int, default=5, help="ensemble size")
    parser.add_argument("--rollout-length", type=int, default=1, help="model rollout horizon")
    parser.add_argument("--rollout-batch", type=int, default=400, help="starting states per rollout")
    parser.add_argument("--model-train-freq", type=int, default=250,
                        help="retrain model every N env steps")
    parser.add_argument("--real-buffer", type=int, default=100_000)
    parser.add_argument("--model-buffer", type=int, default=400_000)
    parser.add_argument("--episodes", type=int, default=200, help="training episodes")
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

    agent = MBPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        lr=args.lr,
        model_lr=args.model_lr,
        gamma=args.gamma,
        tau=args.tau,
        real_buffer_size=args.real_buffer,
        model_buffer_size=args.model_buffer,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_model_members=args.n_members,
        rollout_length=args.rollout_length,
        rollout_batch_size=args.rollout_batch,
        model_train_freq=args.model_train_freq,
        device=device,
    )

    render_env = make_env(args.env, render_mode="human") if args.render_interval else None
    rewards = train(agent, env, args.episodes, render_env=render_env, render_interval=args.render_interval)
    if render_env:
        render_env.close()

    eval_env = make_env(args.env)
    mean_reward = evaluate(agent, eval_env)
    print(f"{args.env} | device: {device} | episodes: {args.episodes} | eval mean reward: {mean_reward:.1f}")

    plot_rewards(rewards, title=f"MBPO on {args.env}")

    env.close()
    eval_env.close()

    if args.watch > 0:
        render_env = make_env(args.env, render_mode="human")
        watch(lambda env: run_episode(agent, env, train=False), render_env, n_episodes=args.watch)
