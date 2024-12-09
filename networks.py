import torch
import torch.nn as nn
from torch.distributions import Categorical 
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()

class ActorCnnAlex(nn.Module):
    
    def __init__(self, env_size, n_actions):
        super(ActorCnnAlex, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3)  # input: [1, 20, 20], output: [32, 18, 18]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                      # output after pool: [32, 9, 9]
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # input: [32, 9, 9], output: [64, 7, 7]
        # After second pooling: [64, 3, 3]
        
        # Fully connected layers
        # Flattened size = 64 * 3 * 3 = 576
        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=n_actions)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()



class ActorCnn(nn.Module):
    
    def __init__(self, env_size, n_actions):
        super(ActorCnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.conv_kernel3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.conv_kernel5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,padding=2)
        self.conv_kernel7 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1,padding=3)
        
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.batch_norm5 = nn.BatchNorm2d(32)
        self.batch_norm7 = nn.BatchNorm2d(32)
        
        self.non_linearity = nn.GELU()
        
        
        self.conv = nn.Conv2d(in_channels=32*3, out_channels=32*3*2, kernel_size=3, stride=1,padding=1)
        
        self.batch_norm = nn.BatchNorm2d(16*3*2)
        
        self.fc = nn.Linear(env_size*env_size*32*3*2, n_actions)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x_ = self.non_linearity(self.conv1(x))
        #x_ = self.batch_norm1(x)
        x3 = self.non_linearity(self.conv_kernel3(x_))
        #x3 = self.batch_norm3(x3)
        x5 = self.non_linearity(self.conv_kernel5(x_))
        #x5 = self.batch_norm5(x5) 
        x7 = self.non_linearity(self.conv_kernel7(x_))
        #x7 = self.batch_norm7(x7)
        x = torch.cat([x3,x5,x7],dim=1)
        x = self.non_linearity((self.conv(x)))
        #x = self.batch_norm(x)
        x = x.flatten(start_dim = 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()
class CriticCnnAlex(nn.Module):
    
    def __init__(self, env_size, n_actions):
            super(CriticCnnAlex, self).__init__()
            
            # First convolutional block
            self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3)  # input: [1, 20, 20], output: [32, 18, 18]
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                      # output after pool: [32, 9, 9]
            
            # Second convolutional block
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # input: [32, 9, 9], output: [64, 7, 7]
            # After second pooling: [64, 3, 3]
            
            # Fully connected layers
            # Flattened size = 64 * 3 * 3 = 576
            self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=n_actions)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

  

    
class CriticCnn(nn.Module):
    
    def __init__(self, env_size, n_actions):
        super(CriticCnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.conv_kernel3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.conv_kernel5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,padding=2)
        self.conv_kernel7 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1,padding=3)
        
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.batch_norm5 = nn.BatchNorm2d(16)
        self.batch_norm7 = nn.BatchNorm2d(16)
        
        self.non_linearity = nn.GELU()
        
        
        self.conv = nn.Conv2d(in_channels=32*3, out_channels=32*3*2, kernel_size=3, stride=1,padding=1)
        
        self.batch_norm = nn.BatchNorm2d(16*3*2)
        
        self.fc = nn.Linear(env_size*env_size*32*3*2, n_actions)
        
    def forward(self, x):
        x_ = self.non_linearity(self.conv1(x))
        #x_ = self.batch_norm1(x)
        x3 = self.non_linearity(self.conv_kernel3(x_))
        #x3 = self.batch_norm3(x3)
        x5 = self.non_linearity(self.conv_kernel5(x_))
        #x5 = self.batch_norm5(x5) 
        x7 = self.non_linearity(self.conv_kernel7(x_))
        #x7 = self.batch_norm7(x7)
        x = torch.cat([x3,x5,x7],dim=1)
        x = self.non_linearity((self.conv(x)))
        #x = self.batch_norm(x)
        x = x.flatten(start_dim = 1)
        x = self.fc(x)
        return x

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)