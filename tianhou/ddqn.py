import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium



import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DeepQNetwork(nn.Module):
    def __init__(self, grid_size=10, n_actions=5):
        super(DeepQNetwork, self).__init__()
        
        # Initial convolution with smaller stride
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual Blocks
        self.res_block1 = ResidualBlock(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.res_block2 = ResidualBlock(64)
        
        # Calculate flattened size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(grid_size, 3, 1), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(grid_size, 3, 1), 3, 1)
        linear_input_size = 2304
        
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res_block2(x)
        
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.4, beta=0.4, beta_increment_per_sampling=0.0001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha    # determines how much prioritization is used
        self.beta = beta      # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.absolute_error_upper = 1.0
        self.epsilon = 0.01   # small amount to avoid zero probabilities

    def add(self, error, sample):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, batch_size):
        batch = []
        batch_indices = []
        batch_priorities = []
        segment = self.tree.total() / batch_size
        beta = np.min([1., self.beta])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            batch_indices.append(idx)
            batch_priorities.append(p)

        sampling_probabilities = batch_priorities / self.tree.total()
        is_weight = np.power(self.tree.count * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        return batch, batch_indices, is_weight

    def update(self, indices, errors):
        for idx, error in zip(indices, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

class DoubleDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DoubleDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Flatten()
            
        )
        
        with torch.no_grad():
            test_input = torch.zeros(1, *input_shape)
            conv_out = self.conv(test_input)
            conv_out_size = conv_out.size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
import torch
import torch.nn as nn
import torch.nn.functional as F



class PrioritizedDoubleQAgent:
    def __init__(self, state_shape, num_actions, learning_rate=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 replay_buffer_size=900000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_network = DeepQNetwork(6, num_actions).to(self.device)
        self.target_network = DeepQNetwork(6, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss(reduction='none')
        
        # Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        
        # Store previous state
        self.previous_state = None
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        if self.previous_state is not None:
            # Initial error is high to ensure early sampling
            if np.random.rand()<0.00001:
                print(f"added")
                print(f"state: {self.previous_state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {done}")
                #raise
            self.replay_buffer.add(1.0, (self.previous_state, action, reward, next_state, done))
        
        self.previous_state = state
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self):
        if self.replay_buffer.tree.count < self.batch_size:
            return
        
        # Sample batch with priorities
        batch, batch_indices, is_weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double Q-learning implementation
        with torch.no_grad():
            # Use online network to select best actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate action values
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss with prioritized experience replay
        td_errors = torch.abs(current_q_values - target_q_values)
        loss = torch.mean(is_weights * self.loss_fn(current_q_values, target_q_values))
        
        # Update priorities
        self.replay_buffer.update(batch_indices, td_errors.cpu().detach().numpy())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_double_dqn(env, num_episodes=10000, update_target_every=20):
    best_reward = -1000
    state_shape = env.observation_space('pred_1').shape
    num_actions = env.action_space('pred_1').n
    
    agent = PrioritizedDoubleQAgent(state_shape, num_actions)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        env.reset(seed=episode)
        agent.previous_state = None
        total_reward = 0
        
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                if agent_name == 'pred_1':
                    action = agent.select_action(observation)
                    
                    if len(env.agents) > 1:
                        agent.store_transition(
                            observation, 
                            action, 
                            reward, 
                            observation, 
                            termination or truncation
                        )
                        agent.train()
                else:
                    action = env.action_space(agent_name).sample()
            
            env.step(action)
            
            if agent_name == 'pred_1':
                total_reward += reward
        
        agent.update_epsilon()
        
        if episode % update_target_every == 0:
            agent.update_target_network()
        
        episode_rewards.append(total_reward)
        
        if episode % 20 == 0:
            print(f"Episode {episode}, Rew : {episode_rewards[-1]} / Avg Reward: {np.mean(episode_rewards[-50:]):3f}, Epsilon: {agent.epsilon:.3f}, Buffer: {agent.replay_buffer.tree.count}")
            if(np.mean(episode_rewards[-10:]) > best_reward and episode > 20):
                best_reward = np.mean(episode_rewards[-10:])
                torch.save(agent.q_network.state_dict(), 'best_model_per.pt')
                print(f'Best model saved with reward {best_reward}')
    return episode_rewards

def main():
    from tianshou_env import raw_env
    env = raw_env()
    rewards = train_double_dqn(env, num_episodes=1000)
    
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title('Training Rewards for pred_1 (PER)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    #save plto to ./
    plt.savefig('PER.png')

if __name__ == '__main__':
    main()