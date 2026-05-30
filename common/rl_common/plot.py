import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards: list[float], title: str, window: int = 100):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.3, label="episode reward")
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), moving_avg, label=f"{window}-ep avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
