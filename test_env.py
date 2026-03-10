import gymnasium as gym
import ale_py
import time

# 1. Register the Arcade Learning Environment (ALE) engine
gym.register_envs(ale_py)

# 2. Create the environment in 'human' render mode
env = gym.make("ALE/Breakout-v5", render_mode="human")

# 3. Reset the environment to get the first frame (state)
observation, info = env.reset()

print(f"Action Space: {env.action_space.n}") 
print(f"Observation Space: {env.observation_space.shape}") 

for step in range(1000):
    # Sample a completely random action
    random_action = env.action_space.sample()
    
    # Step the environment forward
    observation, reward, terminated, truncated, info = env.step(random_action)
    
    # Slow it down slightly so we can watch the rendering
    time.sleep(0.02)
    
    # If the agent loses all lives, reset the game
    if terminated or truncated:
        print("Game Over. Resetting...")
        observation, info = env.reset()

env.close()