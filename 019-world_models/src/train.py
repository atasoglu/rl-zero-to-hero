import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from agent import WorldModelAgent
from rl_common import plot_rewards


def make_car_racing_env(render_mode=None):
    import gymnasium as gym
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=False)
    return env


def collect_random_data(env, n_episodes: int = 20) -> tuple[list, list, list]:
    """Collect frames, actions, rewards with random policy for world model training."""
    all_frames, all_actions, all_rewards = [], [], []
    for _ in tqdm(range(n_episodes), desc="Collecting data", unit="ep"):
        obs, _ = env.reset()
        frames, actions, rewards = [obs], [], []
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(obs)
            actions.append(action)
            rewards.append(float(reward))
        all_frames.append(np.array(frames[:-1]))  # (T, H, W, 3)
        all_actions.append(np.array(actions))
        all_rewards.append(rewards)
    return all_frames, all_actions, all_rewards


def encode_episodes(agent: WorldModelAgent, all_frames: list) -> list:
    """Encode all frames to latent z sequences."""
    all_z = []
    for frames in tqdm(all_frames, desc="Encoding", unit="ep"):
        zs = []
        for frame in frames:
            z, _ = agent.encode(frame)
            zs.append(z.cpu().numpy())
        all_z.append(np.array(zs))
    return all_z


def collect_controller_rollout(agent: WorldModelAgent, env, n_steps: int) -> tuple[dict, float, int]:
    obs, _ = env.reset()
    z_buf, h_buf, actions_buf, log_probs_buf, values_buf, rewards_buf, dones_buf = [], [], [], [], [], [], []
    total_reward = 0.0
    n_episodes = 0

    z, _ = agent.encode(obs)
    h = agent.init_hidden()

    for _ in range(n_steps):
        action, log_prob, value = agent.select_action(z, h)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        z_buf.append(z.cpu().numpy())
        h_buf.append(h.cpu().numpy())
        actions_buf.append(action)
        log_probs_buf.append(log_prob)
        values_buf.append(value)
        rewards_buf.append(float(reward))
        dones_buf.append(float(done))
        total_reward += float(reward)

        h = agent.step_rnn(z, action, h)
        if done:
            obs, _ = env.reset()
            z, _ = agent.encode(obs)
            h = agent.init_hidden()
            n_episodes += 1
        else:
            z, _ = agent.encode(next_obs)

    return {
        "z": np.array(z_buf),
        "h": np.array(h_buf),
        "actions": np.array(actions_buf),
        "log_probs": np.array(log_probs_buf),
        "values": np.array(values_buf),
        "rewards": np.array(rewards_buf),
        "dones": np.array(dones_buf),
    }, total_reward, max(n_episodes, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--z-dim", type=int, default=32, help="VAE latent dimension")
    parser.add_argument("--h-dim", type=int, default=256, help="RNN hidden dimension")
    parser.add_argument("--collect-episodes", type=int, default=20,
                        help="random episodes for world model training")
    parser.add_argument("--vae-epochs", type=int, default=10, help="VAE training epochs")
    parser.add_argument("--rnn-epochs", type=int, default=5, help="MDN-RNN training epochs")
    parser.add_argument("--ctrl-updates", type=int, default=200,
                        help="controller PPO update steps")
    parser.add_argument("--n-steps", type=int, default=256, help="steps per controller rollout")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--watch", type=int, default=3, metavar="N")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    env = make_car_racing_env()
    n_actions = env.action_space.n

    agent = WorldModelAgent(
        action_dim=n_actions,
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        device=device,
    )

    # Stage 1: Collect random data
    print("\n=== Stage 1: Data collection ===")
    all_frames, all_actions, _ = collect_random_data(env, n_episodes=args.collect_episodes)

    # Stage 2: Train VAE
    print("\n=== Stage 2: Training VAE ===")
    flat_frames = np.concatenate(all_frames, axis=0)
    np.random.shuffle(flat_frames)
    vae_loss = agent.train_vae(flat_frames, n_epochs=args.vae_epochs)
    print(f"  VAE loss: {vae_loss:.1f}")

    # Stage 3: Train MDN-RNN
    print("\n=== Stage 3: Training MDN-RNN ===")
    all_z = encode_episodes(agent, all_frames)
    rnn_loss = agent.train_rnn(all_z, all_actions, n_epochs=args.rnn_epochs)
    print(f"  RNN loss: {rnn_loss:.1f}")

    # Stage 4: Train controller
    print("\n=== Stage 4: Training controller ===")
    ctrl_rewards = []
    bar = tqdm(range(args.ctrl_updates), desc="Controller", unit="update")
    for _ in bar:
        rollout, total_reward, n_eps = collect_controller_rollout(agent, env, args.n_steps)
        agent.update_controller(rollout)
        avg_r = total_reward / n_eps
        ctrl_rewards.append(avg_r)
        if len(ctrl_rewards) >= 10:
            bar.set_postfix(reward=f"{avg_r:.1f}", avg10=f"{sum(ctrl_rewards[-10:])/10:.1f}")

    env.close()

    print(f"\nCarRacing-v3 | device: {device} | "
          f"final avg reward: {sum(ctrl_rewards[-10:]) / min(10, len(ctrl_rewards)):.1f}")
    plot_rewards(ctrl_rewards, title="World Models controller on CarRacing-v3")

    if args.watch > 0:
        render_env = make_car_racing_env(render_mode="human")
        for ep in range(args.watch):
            obs, _ = render_env.reset()
            z, _ = agent.encode(obs)
            h = agent.init_hidden()
            done = False
            ep_reward = 0.0
            while not done:
                action, _, _ = agent.select_action(z, h)
                obs, reward, terminated, truncated, _ = render_env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)
                h = agent.step_rnn(z, action, h)
                z, _ = agent.encode(obs)
            print(f"  watch episode {ep + 1}: reward={ep_reward:.1f}")
        render_env.close()
