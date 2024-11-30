import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, observation_space, action_space, learning_rate=0.001, gamma=0.90, 
                 epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01, memory_size=10000, batch_size=64, device=None):
        # Use GPU if available, otherwise use CPU
        self.device = device or (torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu'))
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Memory for experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Neural Network
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.observation_space.shape[1] * self.observation_space.shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space.n)
        )
        return model
    
    def act(self, state):
        # Epsilon-greedy action selection
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_space.n - 1)
        
        # Convert state to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.cpu().argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        # Check if enough memories for a batch
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q-values with target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and update network
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

def train_dqn(env, num_episodes=1000, device=None):
    # Use the provided device or default to CUDA if available
    device = device or (torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu'))
    
    # Create agents for each agent type
    pred_agents = {
        'pred_1': DQNAgent(env.observation_spaces['pred_1'], env.action_spaces['pred_1'], device=device),
        'pred_2': DQNAgent(env.observation_spaces['pred_2'], env.action_spaces['pred_2'], device=device)
    }
    
    prey_agents = {
        'prey_1': DQNAgent(env.observation_spaces['prey_1'], env.action_spaces['prey_1'], device=device),
        'prey_2': DQNAgent(env.observation_spaces['prey_2'], env.action_spaces['prey_2'], device=device)
    }
    
    for episode in range(num_episodes):
        observations = env.reset()
        dones = {agent: False for agent in env.agents}
        episode_reward = {agent: 0 for agent in env.agents}
        
        while not all(dones.values()):
            actions = {}
            
            # Select actions for each agent
            for agent in env.agent_order:
                if not dones[agent]:
                    if 'pred' in agent:
                        actions[agent] = pred_agents[agent].act(observations[agent])
                    else:
                        actions[agent] = prey_agents[agent].act(observations[agent])
            
            # Step environment
            next_observations, rewards, dones, _ = env.step(actions)
            episode_reward = {agent: episode_reward[agent] + rewards[agent] for agent in env.agents}
            # Store experiences and train
            for agent in env.agents:
                if 'pred' in agent:
                    pred_agents[agent].remember(
                        observations[agent], 
                        actions[agent], 
                        rewards[agent], 
                        next_observations[agent], 
                        dones[agent]
                    )
                    pred_agents[agent].replay()
                else:
                    prey_agents[agent].remember(
                        observations[agent], 
                        actions[agent], 
                        rewards[agent], 
                        next_observations[agent], 
                        dones[agent]
                    )
                    prey_agents[agent].replay()
            
            observations = next_observations
        
        # Periodically update target networks
        if episode % 10 == 0:
            for agent in pred_agents:
                pred_agents[agent].update_target_network()
            for agent in prey_agents:
                prey_agents[agent].update_target_network()
        
        # Optional: print progress or performance metrics
        if episode % 2 == 0:
            print(f"Episode {episode} completed")
            print(episode_reward)
            print('epsilon', pred_agents['pred_1'].epsilon)
            
    
    return pred_agents, prey_agents

# Example usage

from env_clean import PettingZooGridWorld
env = PettingZooGridWorld(8,walls=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pred_agents, prey_agents = train_dqn(env, device=device)