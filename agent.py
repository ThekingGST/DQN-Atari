import torch
import torch.nn.functional as F 
import numpy as np
from dqn_network import DQN 
from replay_buffer import ReplayBuffer

class Agent:
    def __init__(
        self,
        n_actions: int,
        device: str = "cpu",
        lr:float = 1e-4,
        gamma: float = 0.99,
        epsilon_start:float = 1.0,
        epsilon_end:float = 0.1,
        epsilon_decay_steps:float = 1_000_000,
        buffer_capacity:int = 100_000,
        batch_size:int = 32,
        target_update_freq:int = 10_000,
        min_buffer_size:int = 10_000,
    ):
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer_size = min_buffer_size

        # --- Epsilon (Exploration Rate) --- 
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        # --- Networks ---
        self.online_net = DQN(n_actions).to(self.device)

        self.target_net = DQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # ---Optimizer---
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # ---Replay Buffer---
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        
        self.steps_done = 0
    
    def select_action(self, state:np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype = torch.float32, device = self.device
            ).unsqueeze(0)

            q_values = self.online_net(state_tensor)

            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end,self.epsilon-self.epsilon_decay)

    def store(self, state, action, reward, next_state,done):
        self.buffer.push(state,action,reward,next_state,done)

    def learn(self):
        if len(self.buffer) < self.min_buffer_size:
            return None 
        
        states,actions,rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states,dtype = torch.float32, device = self.device)
        actions = torch.tensor(actions,dtype = torch.long, device = self.device)
        rewards = torch.tensor(rewards,dtype = torch.float32, device = self.device)
        next_states = torch.tensor(next_states,dtype = torch.float32, device = self.device)
        dones = torch.tensor(dones,dtype = torch.float32, device = self.device)

        current_q = self.online_net(states).gather(dim=1,index=actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]

            target_q = rewards + (1.0 - dones)*self.gamma*max_next_q

        loss = F.smooth_l1_loss(current_q,target_q) 

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(),max_norm=10.0)
        
        self.optimizer.step()

        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.sync_target_network()
        
        return loss.item()
    
    def sync_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())