import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


import torch
import torch.nn as nn
import torch.nn.functional as F

class GridNet(nn.Module):
    def __init__(self, G, num_actions=5):
        super(GridNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
        

        self.batch_norm2 = nn.BatchNorm2d(64)


        # Residual Block
        self.residual_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.residual_bn = nn.BatchNorm2d(64)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_actions)
    
    def forward(self, x):
        # Convolutional Block
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        
        # Residual Block

        x = F.relu(self.residual_bn(self.residual_conv(x))) + x
       

        # Global Average Pooling
        x = self.global_pool(x)  # Shape: (N, 64, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (N, 64)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits (N, num_actions)

        return F.softmax(x, dim=1)  # Convert logits to probabilities


class ImprovedDQNAgent:
    def __init__(self, observation_space, action_space, learning_rate=0.001, gamma=0.9, 
                 epsilon=1.0, epsilon_decay=0.99992, epsilon_min=0.02, memory_size=100000, 
                 batch_size=256, tau=0.003, double_q=True, device=None):
        # Use GPU if available, otherwise use CPUnk
        self.device = device or (torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu'))
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.tau = tau  # Soft update parameter
        self.double_q = double_q  # Double DQN flag
        
        # Memory for prioritized experience replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Two Q-networks to reduce overestimation
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Separate optimizer for each network
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for more stable training
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * self.observation_space.shape[1]//2 * self.observation_space.shape[2]//2, 64),
            nn.GELU(),
            nn.Linear(64, self.action_space.n)
        )
        #return GridNet(self.observation_space.shape[0], self.action_space.n)
        return model
    
    def act(self, state):
        # Adaptive epsilon-greedy with decay
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_space.n - 1)
        
        # Convert state to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.cpu().argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        # Compute TD error for prioritized experience replay
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            current_q = self.model(state_tensor).gather(1, torch.tensor([[action]]).to(self.device)).item()
            next_q = self.target_model(next_state_tensor).max(1)[0].item()
            target_q = reward + (self.gamma * next_q * (1 - done))
            td_error = abs(target_q - current_q)
        
        self.memory.add(td_error, (state, action, reward, next_state, done))
    
    def replay(self):
        # Check if enough memories for a batch
        if len(self.memory) < self.batch_size:
            return
        
        # Sample prioritized batch
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN with prioritized experience replay
        if self.double_q:
            # Use online network to select best action
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(1)
                # Use target network to evaluate the action
                next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            # Standard max Q-learning
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]
        
        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute weighted loss
        loss = (self.loss_fn(current_q_values, target_q_values) * weights).mean()
        
        # Update priorities
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update(indices, td_errors)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)  # Gradient clipping
        self.optimizer.step()
        
        # Decay exploration and update target network
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._soft_update()
    
    def _soft_update(self):
        # Soft update of target network
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used
        self.beta = beta  # To what degree is the sampling biased
        self.beta_increment_per_sampling = beta_increment_per_sampling
        
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.buffer = [None] * capacity
        self.pos = 0
        self.full = False
    
    def add(self, priority, transition):
        # Give max priority to new experience
        index = self.pos if not self.full else np.random.randint(self.capacity)
        
        self.buffer[index] = transition
        self.priorities[index] = priority ** self.alpha
        
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True
    
    def sample(self, batch_size):
        # Prioritize based on priorities
        if self.full:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios / prios.sum()
        indices = np.random.choice(len(prios), batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        total = len(prios)
        weights = (total * probs[indices]) ** -self.beta
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights
    
    def update(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
    
    def __len__(self):
        return self.pos if not self.full else self.capacity

def train_improved_dqn(env, num_episodes=2000, device=None):
    # Use the provided device or default to CUDA if available
    device = device or (torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu'))
    
    # Create agents for each agent type
    pred_agents = {
        'pred_1': ImprovedDQNAgent(env.observation_spaces['pred_1'], env.action_spaces['pred_1'], device=device),
        'pred_2': ImprovedDQNAgent(env.observation_spaces['pred_2'], env.action_spaces['pred_2'], device=device)
    }
    
    prey_agents = {
        'prey_1': ImprovedDQNAgent(env.observation_spaces['prey_1'], env.action_spaces['prey_1'], device=device),
        'prey_2': ImprovedDQNAgent(env.observation_spaces['prey_2'], env.action_spaces['prey_2'], device=device)
    }
    
    # Tracking variables
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Reset environment
        observations = env.reset()
        dones = {agent: False for agent in env.agents}
        episode_reward = {agent: 0 for agent in env.agents}
        
        # Episodic tracking
        steps = 0
        
        while not all(dones.values()):
            steps += 1
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
            
            # Update episode rewards
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
            
        
        
        # Record episode metrics
        episode_rewards.append(sum(episode_reward.values()))
        
        # Logging and visualization
        if episode % 1 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Epsilon: {pred_agents['pred_1'].epsilon:.4f}")
            print(f"  Total Episode Reward: {episode_reward}")

            #if episode_rewards['pred_1'] < -30:
                #save as checkpoint.pt
            
            if episode_reward['pred_1']  > -29:
                print('Solved')
                #save model state dict
                torch.save(pred_agents['pred_1'].model.state_dict(), f"./weights/pred1_{episode}.pt")
                #save target
                torch.save(pred_agents['pred_1'].target_model.state_dict(), f"./weights/target_pred1_{episode}.pt")

                #same for pred_2
                torch.save(pred_agents['pred_2'].model.state_dict(), f"./weights/pred2_{episode}.pt")
                torch.save(pred_agents['pred_2'].target_model.state_dict(), f"./weights/target_pred2_{episode}.pt")

                raise Exception('Solved')
    
    return pred_agents, prey_agents, episode_rewards

# Example usage
from env_clean import PettingZooGridWorld
env = PettingZooGridWorld(8, walls=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pred_agents, prey_agents, rewards = train_improved_dqn(env, device=device)

exit(1)
#-------------------------------------------------------------------------------------------------------------
device = device or (torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu'))

# Create agents for each agent type
pred_agents = {
    'pred_1': ImprovedDQNAgent(env.observation_spaces['pred_1'], env.action_spaces['pred_1'], device=device),
    'pred_2': ImprovedDQNAgent(env.observation_spaces['pred_2'], env.action_spaces['pred_2'], device=device)
}

prey_agents = {
    'prey_1': ImprovedDQNAgent(env.observation_spaces['prey_1'], env.action_spaces['prey_1'], device=device),
    'prey_2': ImprovedDQNAgent(env.observation_spaces['prey_2'], env.action_spaces['prey_2'], device=device)
}

#load policies from /home/bpopper/gtCode/2d_RL_hide_seek/weights/pred1_30.pt
pred_agents['pred_1'].model.load_state_dict(torch.load('/home/bpopper/gtCode/2d_RL_hide_seek/weights/pred1_27.pt'))
pred_agents['pred_1'].target_model.load_state_dict(torch.load('/home/bpopper/gtCode/2d_RL_hide_seek/weights/target_pred1_27.pt'))

pred_agents['pred_2'].model.load_state_dict(torch.load('/home/bpopper/gtCode/2d_RL_hide_seek/weights/pred2_27.pt'))
pred_agents['pred_2'].target_model.load_state_dict(torch.load('/home/bpopper/gtCode/2d_RL_hide_seek/weights/target_pred2_27.pt'))

#set epsilon to 0
pred_agents['pred_1'].epsilon = 0
pred_agents['pred_2'].epsilon = 0

# Tracking variables
episode_rewards = []

for episode in range(1):
    # Reset environment
    observations = env.reset()
    dones = {agent: False for agent in env.agents}
    episode_reward = {agent: 0 for agent in env.agents}
    
    # Episodic tracking
    steps = 0
    
    while not all(dones.values()):
        steps += 1
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
        env.render()
        
        # Update episode rewards
        episode_reward = {agent: episode_reward[agent] + rewards[agent] for agent in env.agents}
        observations = next_observations
    
    # Record episode metrics
    episode_rewards.append(sum(episode_reward.values()))
    
    # Logging and visualization
    if episode % 1 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode}:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Epsilon: {pred_agents['pred_1'].epsilon:.4f}")
        print(f"  Total Episode Reward: {episode_reward}")




