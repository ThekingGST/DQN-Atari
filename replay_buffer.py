import collections
import random
import numpy as np


class ReplayBuffer:
    """
    A fixed-size circular buffer that stores experience tuples
    and allows uniform random sampling for training.
    
    States are stored as uint8 (0-255) to save memory,
    and converted to float32 only when sampled.
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Convert float32 [0, 1] → uint8 [0, 255] to save 4× memory
        state_uint8 = (state * 255).astype(np.uint8)
        next_state_uint8 = (next_state * 255).astype(np.uint8)
        self.buffer.append((state_uint8, action, reward, next_state_uint8, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            # Convert uint8 [0, 255] → float32 [0, 1] only at sample time
            np.array(states, dtype=np.float32) / 255.0,
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32) / 255.0,
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
