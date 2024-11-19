import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pygame
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gym.spaces import Discrete, Box
import time
from typing import Dict, Tuple, List, Optional

# Constants
NUM_POSSIBLE_THINGS = 5
WALL = 0
AIR = 1
SELF = 2  # Changed from SELF_ to SELF for consistency
PREDATOR = 3
PREY = 4

class PettingZooGridWorld(AECEnv):
    def __init__(self, grid_size: int, max_steps: int = 200,walls = True):
        super().__init__()
        self.MAX_STEPS = max_steps
        self.GRID_SIZE = grid_size
        self.NUM_POSSIBLE_THINGS = NUM_POSSIBLE_THINGS
        
        # Initialize pygame attributes
        self.render_window: Optional[pygame.Surface] = None
        self.CELL_SIZE = 50
        self.is_walls = walls
        
        # Initialize environment state
        self.reset()
    
    def _initialize_grid(self) -> None:
        """Initialize the grid with walls and agents."""
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), AIR,dtype=np.int8)
        
        # Add walls
        
        num_walls = self.GRID_SIZE // 2
        wall_positions = set()
        if self.is_walls == False:
            num_walls = 0
        while len(wall_positions) < num_walls:
                x, y = random.randint(0, self.GRID_SIZE-1), random.randint(0, self.GRID_SIZE-1)
                if (x, y) not in wall_positions:
                    wall_positions.add((x, y))
                    self.grid[y, x] = WALL
        
        # Initialize agents
        self.agents = ['prey_1', 'prey_2', 'pred_1', 'pred_2']
        self.agent_positions = {}
        
        # Ensure agents don't spawn on walls or other agents
        occupied_positions = wall_positions.copy()
        for agent in self.agents:
            while True:
                pos = (random.randint(0, self.GRID_SIZE-1), random.randint(0, self.GRID_SIZE-1))
                if pos not in occupied_positions:
                    self.agent_positions[agent] = pos
                    occupied_positions.add(pos)
                    self.grid[pos[1], pos[0]] = PREDATOR if 'pred' in agent else PREY
                    break
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one time step within the environment."""
        self.num_steps += 1
        old_positions = self.agent_positions.copy()
        
        # Move agents
        for agent, action in actions.items():
            x, y = old_positions[agent]
            
            # Movement deltas
            moves = [(1,0), (-1,0), (0,1), (0,-1), (0,0)]
            dx, dy = moves[action]
            
            # Double movement for prey
            if "prey" in agent:
                dx *= 2
                dy *= 2
            
            # Calculate new position with bounds checking
            new_x = max(0, min(self.GRID_SIZE-1, x + dx))
            new_y = max(0, min(self.GRID_SIZE-1, y + dy))
            
            # Only move if destination is not a wall
            if self.grid[new_y, new_x] != WALL:
                self.agent_positions[agent] = (new_x, new_y)
                self.grid[new_y, new_x] = PREDATOR if 'pred' in agent else PREY
            else:
                self.agent_positions[agent] = (x, y)
                self.grid[y, x] = PREDATOR if 'pred' in agent else PREY
        
        return self.get_observations(), self.get_rewards(), self.get_dones(), self.get_infos()
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Return observations for each agent."""
        observations = {}
        for agent, pos in self.agent_positions.items():
            obs = self.grid.copy()
            x, y = pos
            obs[y, x] = SELF
            observations[agent] = obs
        return observations
    
    def get_rewards(self) -> Dict[str, float]:
        """Calculate rewards for each agent."""
        rewards = {agent: 0.0 for agent in self.agents}
        
        for pred in [a for a in self.agents if 'pred' in a]:
            pred_x, pred_y = self.agent_positions[pred]
            
            # Calculate distances to all prey
            distances = []
            for prey in [a for a in self.agents if 'prey' in a]:
                prey_x, prey_y = self.agent_positions[prey]
                manhattan_dist = abs(pred_x - prey_x) + abs(pred_y - prey_y)
                distances.append(manhattan_dist)
            
            min_distance = min(distances)
            rewards[pred] = 1.0 / (min_distance + 1)
            
            # # Bonus reward for catching prey
            # if min_distance == 0:
            #     rewards[pred] += 10.0
        
        return rewards
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment to initial state."""
        self.num_steps = 0
        self._initialize_grid()
        
        # Initialize spaces
        self.action_spaces = {agent: Discrete(5) for agent in self.agents}
        # Box
        self.observation_spaces = {
            agent: Box(
                low=0,
                high=NUM_POSSIBLE_THINGS-1,
                shape=(self.GRID_SIZE, self.GRID_SIZE),
                dtype=np.int8
            ) for agent in self.agents
        }
        
        self.agent_order = self.agents[:]
        self.agent_selector = agent_selector(self.agent_order)
        
        return self.get_observations()
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment."""
        if self.render_window is None:
            pygame.init()
            self.render_window = pygame.display.set_mode((self.GRID_SIZE * self.CELL_SIZE, 
                                                        self.GRID_SIZE * self.CELL_SIZE))
            pygame.display.set_caption('Petting Zoo Grid World')
        
        # Colors
        COLORS = {
            WALL: (0, 0, 0),      # Black
            AIR: (255, 255, 255),  # White
            PREDATOR: (255, 0, 0), # Red
            PREY: (0, 0, 255)      # Blue
        }
        
        self.render_window.fill(COLORS[AIR])
        
        # Draw grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                cell_type = self.grid[y, x]
                if cell_type in COLORS:
                    pygame.draw.rect(
                        self.render_window,
                        COLORS[cell_type],
                        (x * self.CELL_SIZE, y * self.CELL_SIZE, 
                         self.CELL_SIZE, self.CELL_SIZE)
                    )
        
        pygame.display.flip()

    def get_dones(self):

        if self.num_steps >= self.MAX_STEPS:
            self.num_steps = 0
            return {agent: True for agent in self.agents}
        return {agent: False for agent in self.agents}

    def get_infos(self):
        return {agent: {} for agent in self.agents}