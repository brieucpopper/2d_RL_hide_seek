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


NUM_POSSIBLE_THINGS = 5


class PettingZooGridWorld(AECEnv):
    def __init__(self,grid_size,max_steps=1000):
        super().__init__()
        self.MAX_STEPS = max_steps
        self.GRID_SIZE=grid_size
        
        # Initialize grid
        self.grid = [[' ' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # Add walls
        for _ in range(self.GRID_SIZE // 2):
            x, y = random.randint(0, self.GRID_SIZE-1), random.randint(0, self.GRID_SIZE-1)
            self.grid[y][x] = '#'
        
        # Add agents
        self.agents = ['prey_1', 'prey_2', 'pred_1', 'pred_2']
        self.agent_positions = {
            'prey_1': (random.randint(0, self.GRID_SIZE-1), random.randint(0, self.GRID_SIZE-1)),
            'prey_2': (random.randint(0, self.GRID_SIZE-1), random.randint(0, self.GRID_SIZE-1)),
            'pred_1': (random.randint(0, self.GRID_SIZE-1), random.randint(0, self.GRID_SIZE-1)),
            'pred_2': (random.randint(0, self.GRID_SIZE-1), random.randint(0, self.GRID_SIZE-1)),
        }


        
        self.action_spaces = {agent: Discrete(5) for agent in self.agents}
        self.observation_spaces = {agent: Discrete(2*NUM_POSSIBLE_THINGS*self.GRID_SIZE*self.GRID_SIZE) for agent in self.agents}
        
        self.agent_order = self.agents[:]
        #print(f"agent_order: {self.agent_order}")
        self.agent_selector = agent_selector(self.agent_order)
        
        self.render_window = None
        
    def step(self, actions):
        self.num_steps += 1
        
        #print(f"agent: {agent}")
        
        

        #move pred according to actions

        for pred in actions.keys():

            action = actions[pred]
            x, y = self.agent_positions[pred]
            dx, dy = 0, 0

            if action == 0:
                dx = 1
            elif action == 1:
                dx = -1
            elif action == 2:
                dy = 1
            elif action == 3:
                dy = -1
            elif action == 4:
                dx = 0
                dy = 0

            #double dx and dy if "prey" in agent
            if "prey" in pred:
                dx = dx*2
                dy = dy*2

            new_x, new_y = max(0, min(self.GRID_SIZE-1, x + dx)), max(0, min(self.GRID_SIZE-1, y + dy))
            if self.grid[new_y][new_x] != '#':
                self.agent_positions[pred] = (new_x, new_y)

        
        
        # Render the environment
        #self.render()
        
        # Return observations, rewards, dones, infos
        return self.get_observations(), self.get_rewards(), self.get_dones(), self.get_infos()
    
    def render(self, mode='human'):
        if self.render_window is None:
            pygame.init()
            self.render_window = pygame.display.set_mode((self.GRID_SIZE * 50, self.GRID_SIZE * 50))
            pygame.display.set_caption('Petting Zoo Grid World')
        
        self.render_window.fill((255, 255, 255))
        
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == '#':
                    pygame.draw.rect(self.render_window, (0, 0, 0), (x * 50, y * 50, 50, 50))
                else:
                    pygame.draw.rect(self.render_window, (255, 255, 255), (x * 50, y * 50, 50, 50))
        
        for agent, pos in self.agent_positions.items():
            x, y = pos
            if agent.startswith('pred'):
                pygame.draw.circle(self.render_window, (255, 0, 0), (x * 50 + 25, y * 50 + 25), 20)
            else:
                pygame.draw.rect(self.render_window, (0, 0, 255), (x * 50, y * 50, 50, 50))
        
        #pygame.display.flip()
        return pygame.surfarray.array3d(self.render_window)
        pygame.image.save(self.render_window, "env.png")
    
    def get_observations(self):
        observations = {}
        for agent, pos in self.agent_positions.items():
            #wall,air,self, predator (seeker), prey (hider) coded as 0,1,2,3,4
            #a.k.a '#', ' ', '.', 's', 'h' when printed

            isFleeing = 'pred' in agent
            if isFleeing:
                isFleeing=1
            else:
                isFleeing=0

            grid = [0 for _ in range(self.GRID_SIZE*self.GRID_SIZE)]

            for y in range(self.GRID_SIZE):
                for x in range(self.GRID_SIZE):
                    if (x, y) == pos:
                        grid[y*self.GRID_SIZE+x] = 2
                    elif self.grid[y][x] == '#':
                        grid[y*self.GRID_SIZE+x] = 0
                    
                    
                    elif (x, y) == self.agent_positions['prey_1'] or (x, y) == self.agent_positions['prey_2']:
                        grid[y*self.GRID_SIZE+x] = 4
                    elif (x, y) == self.agent_positions['pred_1'] or (x, y) == self.agent_positions['pred_2']:
                        grid[y*self.GRID_SIZE+x] = 3
                    
                    else:
                        grid[y*self.GRID_SIZE+x] = 1
            agent_observation =  grid +[isFleeing]
            #the grid is of size {len(agent_observation)}")
            observations[agent] = agent_observation[:]

        return observations
    
    def get_rewards(self):
        #if pred is in agent name, reward is 1/(distance to closest prey)
        #we just give 0 to preys
        rewards = {agent: 0 for agent in self.agents}
        for agent in self.agents:
            if "pred" in agent:
                x, y = self.agent_positions[agent]
                min_distance = self.GRID_SIZE * 3
                for prey in ['prey_1', 'prey_2']:
                    prey_x, prey_y = self.agent_positions[prey]
                    distance = abs(x - prey_x) + abs(y - prey_y)
                    min_distance = min(min_distance, distance)
                rewards[agent] = 1 / (min_distance+1)
        return rewards
    
    def get_dones(self):

        if self.num_steps >= self.MAX_STEPS:
            self.num_steps = 0
            return {agent: True for agent in self.agents}
        return {agent: False for agent in self.agents}
    
    def get_infos(self):
        return {agent: {} for agent in self.agents}

    def reset(self):
        self.__init__(self.GRID_SIZE)
        self.num_steps = 0
        return self.get_observations()
    

    