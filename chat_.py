import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from pettingzoo.utils import random_demo

# Define SAC Components
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)
        return mean, std


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )


# SAC Hyperparameters
lr = 3e-4
gamma = 0.99
tau = 0.005
alpha = 0.2
hidden_dim = 128
batch_size = 64
buffer_size = 100000
num_epochs = 50
steps_per_epoch = 1000

# Initialize environment
from env_clean import PettingZooGridWorld
env = PettingZooGridWorld(grid_size=5, max_steps=200)
num_agents = len(env.agents)
obs_dim = env.observation_spaces['pred_1'].shape[0] * env.GRID_SIZE * env.GRID_SIZE
action_dim = env.action_spaces['pred_1'].n

# Initialize models
actor = Actor(obs_dim, action_dim, hidden_dim)
critic_1 = Critic(obs_dim + action_dim, hidden_dim)
critic_2 = Critic(obs_dim + action_dim, hidden_dim)
target_critic_1 = Critic(obs_dim + action_dim, hidden_dim)
target_critic_2 = Critic(obs_dim + action_dim, hidden_dim)

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic_1_optimizer = optim.Adam(critic_1.parameters(), lr=lr)
critic_2_optimizer = optim.Adam(critic_2.parameters(), lr=lr)

# Initialize replay buffer
replay_buffer = ReplayBuffer(buffer_size)

# Update target networks
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Training loop
for epoch in range(num_epochs):
    epoch_rewards = []
    for _ in range(steps_per_epoch):
        env.reset()
        done = False
        obs = env.get_observations()
        rewards = {agent: 0 for agent in env.agents}
        
        while not done:
            actions = {}
            
            # Random policy for prey
            for agent in env.agents:
                if "prey" in agent:
                    actions[agent] = env.action_spaces[agent].sample()
                else:
                    state = torch.tensor(obs[agent].flatten(), dtype=torch.float32).unsqueeze(0)
                    mean, std = actor(state)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    actions[agent] = int(action.argmax().item())
            
            # Step environment
            next_obs, rewards, dones, _ = env.step(actions)
            done = all(dones.values())
            for agent in env.agents:
                if "pred" in agent:
                    transition = (
                        obs[agent].flatten(),
                        actions[agent],
                        rewards[agent],
                        next_obs[agent].flatten(),
                        dones[agent],
                    )
                    replay_buffer.add(transition)
            obs = next_obs
            
            epoch_rewards.append(sum(rewards[agent] for agent in env.agents if "pred" in agent))
        
        # Train SAC
        if len(replay_buffer.buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # Update critics
            with torch.no_grad():
                next_actions = actor(next_states)[0].argmax(dim=-1, keepdim=True)
                target_q1 = target_critic_1(torch.cat([next_states, next_actions], dim=-1))
                target_q2 = target_critic_2(torch.cat([next_states, next_actions], dim=-1))
                target_q = rewards + gamma * (1 - dones) * torch.min(target_q1, target_q2)
            
            current_q1 = critic_1(torch.cat([states, actions.unsqueeze(-1)], dim=-1))
            current_q2 = critic_2(torch.cat([states, actions.unsqueeze(-1)], dim=-1))
            critic_1_loss = ((current_q1 - target_q) ** 2).mean()
            critic_2_loss = ((current_q2 - target_q) ** 2).mean()
            
            critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            critic_1_optimizer.step()
            
            critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            critic_2_optimizer.step()
            
            # Update actor
            mean, std = actor(states)
            dist = torch.distributions.Normal(mean, std)
            sampled_actions = dist.sample()
            q1 = critic_1(torch.cat([states, sampled_actions], dim=-1))
            q2 = critic_2(torch.cat([states, sampled_actions], dim=-1))
            actor_loss = -torch.min(q1, q2).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            # Update target critics
            soft_update(target_critic_1, critic_1, tau)
            soft_update(target_critic_2, critic_2, tau)
    
    avg_reward = np.mean(epoch_rewards)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Reward: {avg_reward}")
