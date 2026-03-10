import cv2
import numpy as np
import collections
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self,env, shape=(84,84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,shape[0],shape[1]),
            dtype=np.float32,
        )
    
    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        channel_first = np.expand_dims(normalized, axis=0)
        return channel_first

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = collections.deque(maxlen=n_frames)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_frames, env.observation_space.shape[1], env.observation_space.shape[2]),
            dtype=np.float32,
        )
    
    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self.get_stacked(),info
    
    def observation(self, obs):
        self.frames.append(obs)
        return self.get_stacked()
    
    def get_stacked(self):
        return np.concatenate(list(self.frames), axis=0)

def make_env(env_name="ALE/Breakout-v5"):
    env = gym.make(env_name)
    env = PreprocessFrame(env)
    env = StackFrames(env,n_frames=4)
    return env