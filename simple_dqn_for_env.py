import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import random
import pygame
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gym.spaces import Discrete
import time


def one_hot_encode_state_2d(state,GRID_SIZE, NUM_POSSIBLE_THINGS):
    """Convert grid state to 2D one-hot encoded representation with channels"""
    # Extract the grid state and team indicator
    grid_state = state[:-1]  # All but the last element
    team_indicator = state[-1]  # Last element
    
    # Reshape grid state to 2D
    grid_2d = np.array(grid_state).reshape(GRID_SIZE, GRID_SIZE)
    
    # Create one-hot encoding preserving 2D structure
    # Shape will be (NUM_POSSIBLE_THINGS, GRID_SIZE, GRID_SIZE)
    one_hot_grid = np.zeros((NUM_POSSIBLE_THINGS, GRID_SIZE, GRID_SIZE))
    
    for i in range(NUM_POSSIBLE_THINGS):
        one_hot_grid[i] = (grid_2d == i).astype(np.float32)
    
    return one_hot_grid, team_indicator

class DQN(nn.Module):
    def __init__(self, grid_size, num_channels, action_size):
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate the size of flattened features
        conv_output_size = grid_size * grid_size * 64
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + 1, 512),  # +1 for team indicator
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, grid_state, team_indicator):
        # Process the grid through conv layers
        x = self.conv_layers(grid_state)
        
        # Flatten the conv output
        x = x.view(x.size(0), -1)
        
        # Concatenate with team indicator
        x = torch.cat([x, team_indicator.unsqueeze(1)], dim=1)
        
        # Process through FC layers
        return self.fc_layers(x)
    

class DQNAgent:
    def __init__(self, state_size, action_size,GRID_SIZE, NUM_POSSIBLE_THINGS):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=80000)
        self.gamma = 0.8   # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99992
        self.learning_rate = 0.0005
        self.GRID_SIZE = GRID_SIZE
        self.NUM_POSSIBLE_THINGS = NUM_POSSIBLE_THINGS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(GRID_SIZE, NUM_POSSIBLE_THINGS, action_size).to(self.device)
        self.target_model = DQN(GRID_SIZE, NUM_POSSIBLE_THINGS, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # Store one-hot encoded states in memory
        encoded_state, team_indicator = one_hot_encode_state_2d(state,self.GRID_SIZE, self.NUM_POSSIBLE_THINGS)
        encoded_next_state, next_team_indicator = one_hot_encode_state_2d(state,self.GRID_SIZE, self.NUM_POSSIBLE_THINGS)
        self.memory.append((encoded_state, team_indicator, action, reward, 
                          encoded_next_state, next_team_indicator, done))

    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            encoded_state, team_indicator = one_hot_encode_state_2d(state,self.GRID_SIZE, self.NUM_POSSIBLE_THINGS)
            grid_state = torch.FloatTensor(encoded_state).unsqueeze(0).to(self.device)
            team_indicator = torch.FloatTensor([team_indicator]).to(self.device)
            act_values = self.model(grid_state, team_indicator)
            return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch tensors
        grid_states = torch.FloatTensor(np.array([item[0] for item in minibatch])).to(self.device)
        team_indicators = torch.FloatTensor(np.array([item[1] for item in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([item[2] for item in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([item[3] for item in minibatch])).to(self.device)
        next_grid_states = torch.FloatTensor(np.array([item[4] for item in minibatch])).to(self.device)
        next_team_indicators = torch.FloatTensor(np.array([item[5] for item in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([item[6] for item in minibatch])).to(self.device)

        # Get current Q values
        current_q = self.model(grid_states, team_indicators).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        next_q = self.target_model(next_grid_states, next_team_indicators).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())