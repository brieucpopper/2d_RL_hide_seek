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
NUM_POSSIBLE_THINGS = 4
WALL = 0
SELF = 1
PREDATOR = 2
PREY = 3

def l1_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class PettingZooGridWorld(AECEnv):
    def __init__(self, grid_size: int, max_steps: int = 200,walls = True,num_walls = 5):
        super().__init__()
        self.MAX_STEPS = max_steps
        self.GRID_SIZE = grid_size
        self.NUM_POSSIBLE_THINGS = NUM_POSSIBLE_THINGS
        
        # Initialize pygame attributes
        self.render_window: Optional[pygame.Surface] = None
        self.CELL_SIZE = 100
        self.is_walls = walls
        self.num_walls = num_walls
        
        # Initialize environment state
        self.reset()
    
    def _initialize_grid(self) -> None:
        """Initialize the grid with walls and agents. Grid will be a 3D array of shape (NUM_THINGS ; GRID_SIZE ; GRID SIZE)"""
        self.grid = np.zeros((NUM_POSSIBLE_THINGS, self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        
        # Add walls
        
        num_walls = self.num_walls
        wall_positions = set()

        #add walls all round the grid
        for i in range(self.GRID_SIZE):
            wall_positions.add((0,i))
            wall_positions.add((i,0))
            wall_positions.add((self.GRID_SIZE-1,i))
            wall_positions.add((i,self.GRID_SIZE-1))
            self.grid[WALL,0,i] = 1
            self.grid[WALL,i,0] = 1
            self.grid[WALL,self.GRID_SIZE-1,i] = 1
            self.grid[WALL,i,self.GRID_SIZE-1] = 1

        if self.is_walls == False:
            num_walls = 0
        k = len(wall_positions)
        while len(wall_positions) - k  < num_walls:
                x, y = random.randint(0, self.GRID_SIZE-1), random.randint(0, self.GRID_SIZE-1)
                if (x, y) not in wall_positions:
                    wall_positions.add((x, y))
                    self.grid[WALL,y, x] = 1
        
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
                    if 'pred' in agent:	
                        self.grid[PREDATOR, pos[1], pos[0]] = 1
                    else:
                        self.grid[PREY, pos[1], pos[0]] = 1
                    break
        self.occupied_positions = occupied_positions
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        #self.render()

        #$print(f'actions for this step : {actions}')
       
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
                dx *= 1
                dy *= 1 
            
            # Calculate new position with bounds checking
            new_x = max(0, min(self.GRID_SIZE-1, x + dx))
            new_y = max(0, min(self.GRID_SIZE-1, y + dy))
            
            # Only move if destination is not a wall
            if self.grid[WALL, new_y, new_x] == 0:
                self.agent_positions[agent] = (new_x, new_y)
                
            else:
                self.agent_positions[agent] = (x, y)
        
        #update grid by ftaking out old positions
        for agent, pos in old_positions.items():
            x, y = pos
            if 'pred' in agent:
                self.grid[PREDATOR, y, x] = 0
            else:
                self.grid[PREY, y, x] = 0

        #fill all new positions
        for agent, pos in self.agent_positions.items():
            x, y = pos
            if 'pred' in agent:
                self.grid[PREDATOR, y, x] = 1
            else:
                self.grid[PREY, y, x] = 1
        return self.get_observations(), self.get_rewards(), self.get_dones(), self.get_infos()
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Return observations for each agent."""
        observations = {}
        for agent, pos in self.agent_positions.items():
            obs = self.grid.copy()
            x, y = pos
            obs[SELF, y, x] = 1
            observations[agent] = obs
        return observations
    
    def get_rewards(self) -> Dict[str, float]:
        """Calculate rewards for each agent."""
        rewards = {agent: 0.0 for agent in self.agents}
        
        for pred in [a for a in self.agents if 'pred' in a]:
            ind_agent = pred[5]
            pred_x, pred_y = self.agent_positions[pred]
            #loc_pred = (pred_x, pred_y)
            #loc_prey = {k : v for k,v in self.agent_positions.items() if 'prey' in k}
                    
            #closest_prey = min(loc_prey, key=lambda x: l1_distance(loc_pred, loc_prey[x]))
            #loc_prey_ = loc_prey[closest_prey]
            #distance = l1_distance(loc_pred, loc_prey_)
            
            #rewards[pred] = -distance * 10 #+ 50 * (1 if distance == 0 else 0)
        #for prey in [a for a in self.agents if 'prey' in a]:
            #prey_x, prey_y = self.agent_positions[prey]
            #loc_prey = (prey_x, prey_y)
            #loc_pred = {k : v for k,v in self.agent_positions.items() if 'pred' in k}
            
            #closest_pred = min(loc_pred, key=lambda x: l1_distance(loc_prey, loc_pred[x]))
            #loc_pred_ = loc_pred[closest_pred]
            #distance = l1_distance(loc_prey, loc_pred_)
            
            #rewards[prey] = distance * 10 #- 50 * (1 if distance == 0 else 0)
            
            
            # Calculate distances to prey_1
            prey_1_x, prey_1_y = self.agent_positions[f'prey_{ind_agent}']
            
            rewards[pred] =  -(abs(pred_x - prey_1_x) + abs(pred_y - prey_1_y))/10 + 5 * (1 if (abs(pred_x - prey_1_x) + abs(pred_y - prey_1_y) <= 1) else 0)
            rewards[f'prey_{ind_agent}'] = -1 * rewards[pred]

            #rewards is negative of sum of abs distances
            #if 'pred_1' in pred:
                #rewards['pred_1'] = -1 * (abs(pred_x - prey_1_x) + abs(pred_y - prey_1_y))/10
                #rewards['pred_1'] = -pred_x
            #else:
                #rewards['pred_2'] = -1 * (abs(pred_x - prey_1_x) + abs(pred_y - prey_1_y))/10
                #rewards['pred_2'] = pred_x
        
        #an hidder is hidden if the distance between it and the predator is less than 2
        #All pred get a reward of 1 if at least one prey is not hidden
        '''
        hidden = True
        for pred in [a for a in self.agents if 'pred' in a]:
            pred_x, pred_y = self.agent_positions[pred]
            for prey in [a for a in self.agents if 'prey' in a]:
                prey_x, prey_y = self.agent_positions[prey]
                if abs(pred_x - prey_x) + abs(pred_y - prey_y) <= 2:
                    hidden = False
        if hidden:
            for agent in self.agents:
                if 'prey' in agent:
                    rewards[agent] = 1
                else:
                    rewards[agent] = -1
        else:
            for agent in self.agents:
                if 'prey' in agent:
                    rewards[agent] = -1
                else:
                    rewards[agent] = 1
        '''
        #print('rewards :',rewards)
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
                high=1,
                shape=(NUM_POSSIBLE_THINGS,self.GRID_SIZE, self.GRID_SIZE),
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
            'wall': (0, 0, 0),      # Black
            'air': (255, 255, 255),  # White
            'pred': (255, 0, 0), # Red
            'prey': (0, 0, 255)      # Blue
        }
        
        self.render_window.fill(COLORS['air'])
        
        # Draw gridlines
        for i in range(self.GRID_SIZE):
            pygame.draw.line(self.render_window, (0, 0, 0), (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE))
            pygame.draw.line(self.render_window, (0, 0, 0), (0, i * self.CELL_SIZE), (self.GRID_SIZE * self.CELL_SIZE, i * self.CELL_SIZE))
        



        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[WALL, y, x] == 1:
                    pygame.draw.rect(self.render_window, COLORS['wall'], 
                                     (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
                if self.grid[PREDATOR, y, x] == 1:
                    pygame.draw.rect(self.render_window, COLORS['pred'], 
                                     (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE/2, self.CELL_SIZE))
                if self.grid[PREY, y, x] == 1:
                    #if its prey_1
                    #if (x,y) == self.agent_positions['prey_1']:
                    pygame.draw.rect(self.render_window, COLORS['prey'], 
                                    (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE/2))
                    # pygame.draw.rect(self.render_window, COLORS['prey'], 
                    #                  (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE/2))
                
        
        #pygame.display.flip()
        #time.sleep(0.1)
        #input()
        return pygame.surfarray.array3d(self.render_window)
    def get_dones(self):

        if self.num_steps >= self.MAX_STEPS:
            self.num_steps = 0
            #print('doneeee')
            return {agent: True for agent in self.agents}
        return {agent: False for agent in self.agents}

    def get_infos(self):
        return {agent: {} for agent in self.agents}