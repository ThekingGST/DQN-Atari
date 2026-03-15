import os
import torch
import re
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def extract_metrics(checkpoint_dir="checkpoints"):
    episodes = []
    rewards = []
    
    # Pattern to match dqn_episode_N.pt
    pattern = re.compile(r"dqn_episode_(\d+)\.pt")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Directory '{checkpoint_dir}' not found.")
        return [], [], None
    
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        match = pattern.match(f)
        if match:
            episode_num = int(match.group(1))
            checkpoint_files.append((episode_num, f))
            
    # Sort by episode number
    checkpoint_files.sort()
    
    print(f"Found {len(checkpoint_files)} checkpoints. Extracting data...")
    
    for ep, filename in checkpoint_files:
        path = os.path.join(checkpoint_dir, filename)
        try:
            # We only need 'best_avg_reward', so weights_only=False is needed if checkpoint contains complex objects
            # but we can try to be safe.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            episodes.append(ep)
            rewards.append(checkpoint.get("best_avg_reward", 0.0))
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    # Load best model specifically
    best_data = None
    best_path = os.path.join(checkpoint_dir, "dqn_best.pt")
    if os.path.exists(best_path):
        try:
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_data = {
                "episode": checkpoint.get("episode", 0),
                "reward": checkpoint.get("best_avg_reward", 0.0)
            }
            print(f"Best model found at Episode {best_data['episode']} with Reward {best_data['reward']:.2f}")
        except Exception as e:
            print(f"Warning: Could not load dqn_best.pt: {e}")

    return episodes, rewards, best_data

def plot_progress(episodes, rewards, best_data, output_file="training_progress.png"):
    if not episodes:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='b', label="Average Reward")
    
    if best_data:
        plt.scatter(best_data["episode"], best_data["reward"], color='red', s=100, label="DQN Best", zorder=5)
        plt.annotate(f"Best: {best_data['reward']:.2f}\n(Ep {best_data['episode']})",
                     (best_data["episode"], best_data["reward"]),
                     textcoords="offset points", xytext=(0, 10), ha='center',
                     color='red', weight='bold')

    plt.title("DQN Atari Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Best Average Reward (Last 100)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    eps, rews, best = extract_metrics()
    plot_progress(eps, rews, best)
