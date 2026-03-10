import os
import numpy as np
import torch
from collections import deque

from preprocessing import make_env
from agent import Agent


def train(
    env_name: str = "ALE/Breakout-v5",
    num_episodes: int = 10_000,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 20,
    save_interval: int = 200,
    snap_interval:int = 1000,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    env = make_env(env_name)
    n_actions = env.action_space.n
    print(f"Environment: {env_name} | Actions: {n_actions}")

    agent = Agent(
        n_actions=n_actions,
        device=device,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=1_000_000,
        buffer_capacity=100_000,
        batch_size=32,
        target_update_freq=10_000,
        min_buffer_size=10_000,
    )

    reward_history = deque(maxlen=100)
    best_avg_reward = -float("inf")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Resume from checkpoint if it exists ---
    start_episode = 1
    total_steps = 0
    resume_path = os.path.join(checkpoint_dir, "dqn_latest.pt")

    if os.path.exists(resume_path):
        checkpoint = load_checkpoint(agent, resume_path)
        start_episode = checkpoint["episode"] + 1
        total_steps = checkpoint["total_steps"]
        best_avg_reward = checkpoint["best_avg_reward"]

        # Restore reward history
        saved_history = checkpoint.get("reward_history", [])
        for r in saved_history:
            reward_history.append(r)

        print(
            f"Resumed from Episode {start_episode - 1} | "
            f"Steps: {total_steps} | "
            f"ε: {agent.epsilon:.4f} | "
            f"Best Avg: {best_avg_reward:.2f}"
        )
    else:
        print("Starting fresh — no checkpoint found.")

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

    for episode in range(start_episode, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        episode_loss = []
        done = False

        while not done:
            # 1. Select action (ε-greedy)
            action = agent.select_action(state)

            # 2. Step the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. Clip reward to {-1, 0, +1}
            clipped_reward = float(np.sign(reward))

            # 4. Store transition in replay buffer
            agent.store(state, action, clipped_reward, next_state, done)

            # 5. Learn from a random batch
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            # 6. Decay epsilon
            agent.decay_epsilon()

            # 7. Move to next state
            state = next_state
            episode_reward += reward  # Track UNCLIPPED reward for logging
            total_steps += 1

        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0

        if episode % log_interval == 0:
            print(
                f"Episode {episode:>6} | "
                f"Reward: {episode_reward:>7.1f} | "
                f"Avg(100): {avg_reward:>7.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Steps: {total_steps:>8} | "
                f"Buffer: {len(agent.buffer):>6}"
            )

        # --- Save latest checkpoint (for resuming) ---
        if episode % snap_interval == 0:
            # Save numbered checkpoint
            path = os.path.join(checkpoint_dir, f"dqn_episode_{episode}.pt")
            save_checkpoint(agent, path, episode, total_steps, best_avg_reward, reward_history)

        if episode % save_interval == 0:
            # Always overwrite "latest" so resume picks up the most recent state
            latest_path = os.path.join(checkpoint_dir, "dqn_latest.pt")
            save_checkpoint(agent, latest_path, episode, total_steps, best_avg_reward, reward_history)

        # --- Save best model ---
        if avg_reward > best_avg_reward and len(reward_history) == 100:
            best_avg_reward = avg_reward
            path = os.path.join(checkpoint_dir, "dqn_best.pt")
            save_checkpoint(agent, path, episode, total_steps, best_avg_reward, reward_history)
            print(f"  ★ New best average reward: {best_avg_reward:.2f} — model saved!")

    print("\nTraining complete!")
    print(f"Best average reward (last 100 episodes): {best_avg_reward:.2f}")

    env.close()
    return agent


# ------------------------------------------------------------------
# Checkpoint Utilities
# ------------------------------------------------------------------

def save_checkpoint(agent, path, episode, total_steps, best_avg_reward, reward_history):
    """Save everything needed to fully resume training."""
    torch.save(
        {
            # Network & optimizer state
            "online_net": agent.online_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "optimizer": agent.optimizer.state_dict(),

            # Agent state
            "epsilon": agent.epsilon,
            "steps_done": agent.steps_done,

            # Training loop state
            "episode": episode,
            "total_steps": total_steps,
            "best_avg_reward": best_avg_reward,
            "reward_history": list(reward_history),
        },
        path,
    )


def load_checkpoint(agent, path):
    """Load a checkpoint and restore agent state. Returns the full checkpoint dict."""
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    agent.online_net.load_state_dict(checkpoint["online_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    agent.epsilon = checkpoint["epsilon"]
    agent.steps_done = checkpoint["steps_done"]
    print(f"Checkpoint loaded from {path}")
    return checkpoint


if __name__ == "__main__":
    train()