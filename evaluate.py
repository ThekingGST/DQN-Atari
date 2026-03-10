import numpy as np
import torch
import gymnasium as gym

from preprocessing import make_env
from agent import Agent
from train import load_checkpoint


def evaluate(
    env_name: str = "ALE/Breakout-v5",
    checkpoint_path: str = "checkpoints/dqn_best.pt",
    num_episodes: int = 5,
    render: bool = True,
):
    """
    Watch the trained agent play.

    Args:
        env_name:         Gymnasium Atari environment name
        checkpoint_path:  Path to saved model weights
        num_episodes:     Number of games to play
        render:           Whether to render the game visually
    """
    # Create environment (with rendering)
    if render:
        raw_env = gym.make(env_name, render_mode="human")
    else:
        raw_env = gym.make(env_name)

    # Apply the same preprocessing wrappers
    from preprocessing import PreprocessFrame, StackFrames
    env = StackFrames(PreprocessFrame(raw_env), n_frames=4)

    n_actions = env.action_space.n

    # Create agent and load trained weights
    agent = Agent(n_actions=n_actions, device="cpu")
    load_checkpoint(agent, checkpoint_path)

    # Set epsilon to near-zero (almost no exploration)
    agent.epsilon = 0.05

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

        print(f"Episode {episode} | Reward: {episode_reward:.1f}")

    env.close()


if __name__ == "__main__":
    evaluate()