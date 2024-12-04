import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete,Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import pygame
import time
import random

NUM_POSSIBLE_THINGS = 5
WALL = 0
PRED_1 = 1
PRED_2 = 2
HIDER_1 = 3
HIDER_2 = 4


NUM_ITERS = 100


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "Hide_Seek_2d"}

    def __init__(self, render_mode=None, grid_size=6):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["pred_1", "pred_2", "hider_1", "hider_2"]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        #map is NUMTHINGS x Grid_size x Grid_size 
        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(5) for agent in self.possible_agents}
        self.CELL_SIZE = 100
        self._observation_spaces = {
            agent: Box(low=0, high=1,shape=(NUM_POSSIBLE_THINGS,grid_size,grid_size),dtype=np.int8) for agent in self.possible_agents
        }
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.NUM_POSSIBLE_THINGS = NUM_POSSIBLE_THINGS

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self._observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        
        if self.render_mode == "human":
            pygame.init()
            self.render_window = pygame.display.set_mode((self.grid_size * self.CELL_SIZE, 
                                                        self.grid_size * self.CELL_SIZE))
            pygame.display.set_caption('Petting Zoo Grid World')
        
            # Colors
            COLORS = {
                'wall': (0, 0, 0),      # Black
                'air': (255, 255, 255),  # White
                'pred_1': (255, 60, 0),    
                'pred_2': (255, 0, 60), 
                'hider_1': (30, 0, 255),   
                'hider_2': (0, 60, 200) 
            }
            
            self.render_window.fill(COLORS['air'])
            
            # Draw gridlines
            for i in range(self.grid_size):
                pygame.draw.line(self.render_window, (0, 0, 0), (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.grid_size * self.CELL_SIZE))
                pygame.draw.line(self.render_window, (0, 0, 0), (0, i * self.CELL_SIZE), (self.grid_size * self.CELL_SIZE, i * self.CELL_SIZE))
            



            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.grid_for_display[WALL, y, x] == 1:
                        pygame.draw.rect(self.render_window, COLORS['wall'], 
                                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
                    if self.grid_for_display[PRED_1, y, x] == 1:
                        pygame.draw.rect(self.render_window, COLORS['pred_1'], 
                                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
                    if self.grid_for_display[PRED_2, y, x] == 1:
                        pygame.draw.rect(self.render_window, COLORS['pred_2'], 
                                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
                    if self.grid_for_display[HIDER_1, y, x] == 1:
                        pygame.draw.rect(self.render_window, COLORS['hider_1'], 
                                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
                    if self.grid_for_display[HIDER_2, y, x] == 1:
                        pygame.draw.rect(self.render_window, COLORS['hider_2'], 
                                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
            
            pygame.display.flip()
            input()

        

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.render_mode == "human":
            pygame.quit()
        pass

    def generate_grid(self,grid_size):
        '''
        Generates a grid of size grid_size x grid_size with walls on the edges and predator and prey in the corners
        returns this grid
        populates the self.infos dict with the coords of the agents
        '''

        grid = np.zeros((NUM_POSSIBLE_THINGS,grid_size,grid_size),dtype=np.int8)

        #populate walls on the edges
        for edge_idx in range(grid_size):
            grid[WALL,edge_idx,0] = 1
            grid[WALL,edge_idx,grid_size-1] = 1
            grid[WALL,0,edge_idx] = 1
            grid[WALL,grid_size-1,edge_idx] = 1
        

        self.infos["pred_1"]["coords"] = [random.randint(1,grid_size-2),random.randint(1,grid_size-2)]
        self.infos["pred_2"]["coords"] = [random.randint(1,grid_size-2),random.randint(1,grid_size-2)]
        self.infos["hider_1"]["coords"] = [random.randint(1,grid_size-2),random.randint(1,grid_size-2)]
        self.infos["hider_2"]["coords"] = [random.randint(1,grid_size-2),random.randint(1,grid_size-2)]

        grid = self.write_grid_from_info(grid,self.infos,"pred_1")
        return grid
    
    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents} #use this dict for coords
        self.grid_for_display = self.generate_grid(self.grid_size) #will populate agent coords
        self.state = {agent: self.grid_for_display for agent in self.agents} #TODO for now state and obs are equal
        self.observations = {agent: self.grid_for_display for agent in self.agents} #TODO for now state and obs are equal
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()


    def seed(self, seed):
        np.random.seed(seed)

    def get_step_after_action(self,grid_for_agent,action):
        '''
        Returns the grid after the action is taken
        updates grid and agent coords
        '''

        agent_coords_before = self.infos[self.agent_selection]["coords"].copy()
        #make sure self.grid
        
        #visually 3 : bas
        # 1 droite

        # 4: haut

        # 2 gauche

        # 0 : pas bouger

        actions = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
        
        potential_new_coords = agent_coords_before + np.array(actions[action],dtype=np.int8)
        #clamp the potential_new_coords in the grid
        potential_new_coords = np.clip(potential_new_coords,0,self.grid_size-1)

        #check if the move is valid
        if grid_for_agent[WALL,potential_new_coords[0],potential_new_coords[1]] == 0:
            #move is valid
            self.infos[self.agent_selection]["coords"] = potential_new_coords
        else:
            self.infos[self.agent_selection]["coords"] = agent_coords_before

        return self.write_grid_from_info(grid_for_agent,self.infos,self.agent_selection)


    def write_grid_from_info(self,grid_to_change,info,agent_self):
        #write zeros to grid_to_change[PRED]
        grid_to_change[PRED_1] = 0
        grid_to_change[PRED_2] = 0

        grid_to_change[HIDER_1] = 0
        grid_to_change[HIDER_2] = 0

        for agent in self.agents:
            if agent == 'pred_1':
                grid_to_change[PRED_1,info[agent]["coords"][0],info[agent]["coords"][1]] = 1
            elif agent == 'pred_2':
                grid_to_change[PRED_2,info[agent]["coords"][0],info[agent]["coords"][1]] = 1
            elif agent == 'hider_1':
                grid_to_change[HIDER_1,info[agent]["coords"][0],info[agent]["coords"][1]] = 1
            else:
                grid_to_change[HIDER_2,info[agent]["coords"][0],info[agent]["coords"][1]] = 1

        return grid_to_change
    

    def compute_rewards_all_agents(self):

        #for now the rewards are 0 for all hiders, and x+y for all predators

        rewards = {}

        target_x = self.infos["hider_1"]["coords"][0]
        target_y = self.infos["hider_1"]["coords"][1]

        p1x = self.infos["pred_1"]["coords"][0]
        p1y = self.infos["pred_1"]["coords"][1]

    
        for agent in self.agents:
            if agent.startswith("pred_1"):
                rewards[agent] = -abs(p1x - target_x)
                #rewards[agent] = p1x
            elif agent.startswith("pred_2"):
                #rewards[agent] = self.infos["pred_2"]["coords"][0]
                rewards[agent] = 0
            else:
                rewards[agent] = 0
        #$print(rewards)
        return rewards





    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations DONE
        - truncations DONE
        - infos
        - agent_selection (to the next agent)
        - 
        And any internal state used by observe() or render()
        #dont forget to update GRID and AGENT COORDS
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            action = None
            self._was_dead_step(action)
            return
        
        
        agent = self.agent_selection
        
        
        # apply the action to the game
        self.state[self.agent_selection] =  self.get_step_after_action(self.state[agent], action) #updates grid and agent coords
        self.observations[self.agent_selection] = self.state[self.agent_selection].copy() #TODO for now state and obs are equal
        
        self._cumulative_rewards[agent] = 0
        #print(f' agent {agent} took action {action} and got observation {self.observations[agent]}')
        # collect reward if it is the last agent to act
        self.grid_for_display = self.state[self.agent_selection].copy()
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # compute rewards FOR ALL AGENTS with syntax self.rewards[self.agents[0]]

            #update grid
            

            self.rewards = self.compute_rewards_all_agents()
            

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

            
        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            print(f' agent {agent} took action {action}')
            print(self._cumulative_rewards)
            self.render()
        
        #print cum rew
        #print(f'cumulative rewards {self._cumulative_rewards}')


        
        





def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode,grid_size=6)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# from pettingzoo.test import api_test




# env = env(render_mode="human")
# env.reset(seed=42)

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()

#     if termination or truncation:
#         action = None
#     else:
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample()

#     env.step(action)
# env.close()