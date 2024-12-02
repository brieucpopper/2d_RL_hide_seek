import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Flatten()
        )
        
        # Compute the size of the flattened layer
        with torch.no_grad():
            test_input = torch.zeros(1, *input_shape)
            conv_out = self.conv(test_input)
            conv_out_size = conv_out.size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 replay_buffer_size=900000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Network
        self.q_network = DQN(state_shape, num_actions).to(self.device)
        self.target_network = DQN(state_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
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
            self.replay_buffer.append((self.previous_state, action, reward, next_state, done))
            # print(f'self previous state is {self.previous_state[1]}')
            # print(f'state is {state[1]}')
            # print(f'action is {action}')
            # print(f'reward is {reward}')
        
        self.previous_state = state
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(env, num_episodes=5000, update_target_every=20):

    best_reward = -1000
    # Initialize DQN agent for pred_1
    state_shape = env.observation_space('pred_1').shape
    num_actions = env.action_space('pred_1').n
    
    agent = DQNAgent(state_shape, num_actions)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        env.reset(seed=episode)
        agent.previous_state = None  # Reset previous state for each episode
        total_reward = 0
        
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                if agent_name == 'pred_1':
                    # Use DQN agent for pred_1
                    action = agent.select_action(observation)
                    
                    # Store transition for learning
                    if len(env.agents) > 1:  # Ensure we have a previous state to learn from
                        agent.store_transition(
                            observation, 
                            action, 
                            reward, 
                            observation, 
                            termination or truncation
                        )

                        agent.train()
                else:
                    # Random action for other agents
                    action = env.action_space(agent_name).sample()
            
            env.step(action)
            
            if agent_name == 'pred_1':
                total_reward += reward
        
        # Decay epsilon
        agent.update_epsilon()
        
        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_network()
        
        episode_rewards.append(total_reward)
        
        if episode % 4 == 0:
            print(f"Episode {episode}, Rew : {episode_rewards[-1]} / Avg Reward: {np.mean(episode_rewards[-50:]):3f}, Epsilon: {agent.epsilon:.3f}, Buffer: {len(agent.replay_buffer)}")
            if(np.mean(episode_rewards[-10:]) > best_reward and episode > 20):
                best_reward = np.mean(episode_rewards[-10:])
                torch.save(agent.q_network.state_dict(), 'best_model_squared.pt')
                print(f'Best model saved with reward {best_reward}')
    return episode_rewards

def main():
    # Create environment
    from tianshou_env import raw_env

    env = raw_env()
    
    # Train DQN
    rewards = train_dqn(env, num_episodes=1000)
    
    # Optional: plot rewards
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title('Training Rewards for pred_1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def watch_play_from_ckpt(path):
    from tianshou_env import raw_env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=raw_env(render_mode = "human")

    network = DQN(env.observation_space('pred_1').shape, env.action_space('pred_1').n).to(device)
    network.load_state_dict(torch.load(path))

    state_shape = env.observation_space('pred_1').shape
    num_actions = env.action_space('pred_1').n
    
    agent = DQNAgent(state_shape, num_actions)

    agent.q_network = network
    agent.target_network = network
    #set eps to 0
    agent.epsilon = 0
    

    
    for episode in range(1):
        env.reset(seed=episode)
        agent.previous_state = None  # Reset previous state for each episode
        total_reward = 0
        episode_rewards = []
        
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                if agent_name == 'pred_1':
                    # Use DQN agent for pred_1
                    action = agent.select_action(observation)
                    
                else:
                    # Random action for other agents
                    action = env.action_space(agent_name).sample()
            
            env.step(action)
            
            if agent_name == 'pred_1':
                total_reward += reward
        
        episode_rewards.append(total_reward)
        
        
        print(f"Episode {episode}, Rew : {episode_rewards[-1]} / Avg Reward: {np.mean(episode_rewards[-50:]):3f}, Epsilon: {agent.epsilon:.3f}, Buffer: {len(agent.replay_buffer)}")


if __name__ == '__main__':

    #watch_play_from_ckpt('/home/hice1/bpopper3/scratch/2d_RL_hide_seek/tianhou/best_model_squared.pt')
    main()




