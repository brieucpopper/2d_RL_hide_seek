import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete,Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

import pygame
import time
import random

NUM_POSSIBLE_THINGS = 6
WALL = 0
PRED_1 = 1
PRED_2 = 2
HIDER_1 = 3
HIDER_2 = 4
MOVABLE_WALL = 5


NUM_ITERS = 200

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "2dhs"}

    def __init__(self, render_mode=None,grid_size=6,walls=False,generate_gif=False):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["pred_1", "pred_2", "hider_1", "hider_2"]
        self.grid_size = grid_size
        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.CELL_SIZE = 90
        self.walls = walls #true or false
        if(self.walls):
            print("Using WALLS")
        self.generate_gif = generate_gif
        self.gif_frames = []
        self.IS_SCREEN = False

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=0, high=1,shape=(NUM_POSSIBLE_THINGS,self.grid_size,self.grid_size),dtype=np.int8)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
    
    def render(self):
        """
        Renders the environment. In human mode, it renders to a pygame surface.
        Optionally opens a graphical window if IS_SCREEN is True.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "human":
            import pygame
            import numpy as np
            import imageio

            pygame.init()

            # Initialize rendering surface
            if not hasattr(self, 'render_surface'):
                self.render_surface = pygame.Surface(
                    (self.grid_size * self.CELL_SIZE, self.grid_size * self.CELL_SIZE)
                )

            # Colors
            COLORS = {
                'wall': (0, 0, 0),        # Black
                'air': (255, 255, 255),  # White
                'pred_1': (255, 60, 0),
                'pred_2': (255, 0, 60),
                'hider_1': (120, 0, 255),
                'hider_2': (0, 120, 200),
                'movable_wall': (139, 69, 19)  # Brown
            }

            self.render_surface.fill(COLORS['air'])

            # Draw gridlines
            for i in range(self.grid_size):
                pygame.draw.line(self.render_surface, (0, 0, 0), (i * self.CELL_SIZE, 0),
                                (i * self.CELL_SIZE, self.grid_size * self.CELL_SIZE))
                pygame.draw.line(self.render_surface, (0, 0, 0), (0, i * self.CELL_SIZE),
                                (self.grid_size * self.CELL_SIZE, i * self.CELL_SIZE))
                
            #blit coords
            font = pygame.font.Font(None, 5)
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    text = font.render(f"{x},{y}", True, (0, 0, 0))
                    self.render_surface.blit(text, (x * self.CELL_SIZE + 5, y * self.CELL_SIZE + 5))

            # Draw objects on the grid
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.grid[PRED_1, y, x] == 1:
                        sprite = pygame.image.load('./PARALLEL/sprites/pred_1.png')
                        self.render_surface.blit(sprite, (x * self.CELL_SIZE + 20, y * self.CELL_SIZE + 50))

                    if self.grid[PRED_2, y, x] == 1:
                        sprite = pygame.image.load('./PARALLEL/sprites/pred_2.png')
                        self.render_surface.blit(sprite, (x * self.CELL_SIZE + 20, y * self.CELL_SIZE + 20))

                    if self.grid[HIDER_1, y, x] == 1:
                        sprite = pygame.image.load('./PARALLEL/sprites/hider_1.png')
                        self.render_surface.blit(sprite, (x * self.CELL_SIZE + 50, y * self.CELL_SIZE + 50))

                    if self.grid[HIDER_2, y, x] == 1:
                        sprite = pygame.image.load('./PARALLEL/sprites/hider_2.png')
                        self.render_surface.blit(sprite, (x * self.CELL_SIZE + 50, y * self.CELL_SIZE + 20))

                    if self.grid[WALL, y, x] == 1:
                        pygame.draw.rect(self.render_surface, COLORS['wall'],
                                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

                    if self.grid[MOVABLE_WALL, y, x] == 1:
                        pygame.draw.rect(self.render_surface, COLORS['movable_wall'],
                                        (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

                    # Add text labels for debugging
                    font = pygame.font.Font(None, 14)
                    if self.grid[PRED_1, y, x] == 1:
                        text = font.render("P1", True, (0, 0, 0))
                        self.render_surface.blit(text, (x * self.CELL_SIZE + 30, y * self.CELL_SIZE + 80))

                    if self.grid[PRED_2, y, x] == 1:
                        text = font.render("P2", True, (0, 0, 0))
                        self.render_surface.blit(text, (x * self.CELL_SIZE + 30, y * self.CELL_SIZE + 10))

                    if self.grid[HIDER_1, y, x] == 1:
                        text = font.render("H1", True, (0, 0, 0))
                        self.render_surface.blit(text, (x * self.CELL_SIZE + 60, y * self.CELL_SIZE + 80))

                    if self.grid[HIDER_2, y, x] == 1:
                        text = font.render("H2", True, (0, 0, 0))
                        self.render_surface.blit(text, (x * self.CELL_SIZE + 60, y * self.CELL_SIZE + 10))

                    if self.grid[MOVABLE_WALL, y, x] == 1:
                        text = font.render("WALL", True, (0, 0, 0))
                        self.render_surface.blit(text, (x * self.CELL_SIZE, y * self.CELL_SIZE + 70))

            # Handle screen display or GIF generation
            if self.IS_SCREEN:
                if not hasattr(self, 'render_window'):
                    self.render_window = pygame.display.set_mode(
                        (self.grid_size * self.CELL_SIZE, self.grid_size * self.CELL_SIZE))
                    pygame.display.set_caption('Petting Zoo Grid World')

                self.render_window.blit(self.render_surface, (0, 0))
                pygame.display.flip()
            elif self.generate_gif:
                if not hasattr(self, 'gif_frames'):
                    self.gif_frames = []

                if len(self.gif_frames) < 90:
                    frame = pygame.surfarray.array3d(self.render_surface)
                    frame = np.transpose(frame, (1, 0, 2))  # Transpose for correct orientation
                    self.gif_frames.append(frame)
                else:
                    path = './random.gif'
                    imageio.mimsave(path, self.gif_frames, fps=2)
                    print(f"saved GIF at path {path}")
                    exit(1)


    def close(self):
        pass

    def reset(self,seed=None,options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        
        self.infos = {agent: {} for agent in self.agents}

        self.grid = self.generate_grid(self.grid_size)
        observations = {agent: self.grid for agent in self.agents}
        self.state = observations
        return observations, self.infos
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

        ################################################################
        #place 4 walls randomly
        # if self.walls:
        #     #add a movable wall in the middle
        #     grid[MOVABLE_WALL,grid_size//2,grid_size//2] = 1

        #     for _ in range(4):
        #         potential_x = random.randint(1,grid_size-2)
        #         potential_y = random.randint(1,grid_size-2)

        #         if grid[MOVABLE_WALL,potential_x,potential_y] == 0:
        #             grid[WALL,potential_x,potential_y] = 1
            
        #custom map with movable wall that can be used to block
        ################################################################



        if self.walls:
            
            #add a movable wall in the middle
            grid[MOVABLE_WALL,2,3] = 1

            # for _ in range(0):
            #     potential_x = random.randint(1,grid_size-2)
            #     potential_y = random.randint(1,grid_size-2)

            #     if grid[MOVABLE_WALL,potential_x,potential_y] == 0:
            #         grid[WALL,potential_x,potential_y] = 1]

            #walls at y = 4, all x except 1
            for x in range(1,grid_size-1):
                grid[WALL,3,x] = 1
            
            grid[WALL,3,3] = 0

                
            
        
        

        self.infos["pred_1"]["coords"] = [random.randint(1,grid_size-2),random.randint(1,grid_size-2)]
        self.infos["pred_2"]["coords"] = [random.randint(1,grid_size-2),random.randint(1,grid_size-2)]
        self.infos["hider_1"]["coords"] = [random.randint(1,grid_size-2),random.randint(1,grid_size-2)]
        self.infos["hider_2"]["coords"] = [random.randint(1,grid_size-2),random.randint(1,grid_size-2)]

        self.infos["pred_1"]["coords"] = [4,5]
        self.infos["pred_2"]["coords"] = [5,1]
        self.infos["hider_1"]["coords"] = [2,4]
        self.infos["hider_2"]["coords"] = [1,1]
        
        

        grid = self.write_grid_from_info(grid,self.infos)
        return grid
    def compute_rewards_all_agents(self):

        #for now the rewards are 0 for all hiders, and x+y for all predators

        rewards = {}

        target_x = self.infos["hider_1"]["coords"][0]
        target_y = self.infos["hider_1"]["coords"][1]

        p1x = self.infos["pred_1"]["coords"][0]
        p1y = self.infos["pred_1"]["coords"][1]

        hider2x = self.infos["hider_2"]["coords"][0]
        hider2y = self.infos["hider_2"]["coords"][1]

        p2x = self.infos["pred_2"]["coords"][0]

        hider2x = self.infos["hider_2"]["coords"][0]
        hider2y = self.infos["hider_2"]["coords"][1]

    
        for agent in self.agents:
            if agent.startswith("pred_1"):
                
                rewards[agent] = -(np.abs(p1x-target_x)**2 + np.abs(p1y-target_y)**2)
                #rewards[agent] = p1x + p1y
            elif agent.startswith("pred_2"):
                #rewards[agent] = self.infos["pred_2"]["coords"][0]
                rewards[agent] = 0
            elif agent.startswith("hider_1"):
                rewards[agent] = (np.abs(p1x-target_x)**2 + np.abs(p1y-target_y)**2)
            else:
                rewards[agent] = 0
        return rewards
    

    def write_grid_from_info(self,grid_to_change,info):
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
    
    def apply_actions(self,actions):

        ACTIONS = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
        movable_wall_moved = False #ensure only one agent can move the wall
        for agent in self.agents:
            coord_be4 = self.infos[agent]["coords"]
            action = actions[agent]

            dx,dy = ACTIONS[action]
            potential_new_coords = coord_be4 + np.array([dx,dy],dtype=np.int8)
            potential_new_coords = np.clip(potential_new_coords,0,self.grid_size-1)

            isValidMove = None

            
            if self.grid[WALL,potential_new_coords[0],potential_new_coords[1]] == 1:
                isValidMove = False # Trying to move in a wall
            else:
                if self.grid[MOVABLE_WALL,potential_new_coords[0],potential_new_coords[1]] == 1:
                    # There is a movable wall
                    # check if the wall can be moved 
                    if movable_wall_moved:
                        isValidMove = False # The wall was already moved by an agent
                    else:
                        #the wall can be moved IF THERE ISNT A WALL BEHIND OR A PLAYER's old position (you can defend a wall this way)
                        new_wall_coords = coord_be4 + np.array([2*dx,2*dy],dtype=np.int8)
                        new_wall_coords = np.clip(new_wall_coords,0,self.grid_size-1)
                        if self.grid[WALL,new_wall_coords[0],new_wall_coords[1]] == 1 or self.grid[PRED_1,new_wall_coords[0],new_wall_coords[1]] == 1 or self.grid[PRED_2,new_wall_coords[0],new_wall_coords[1]] == 1 or self.grid[HIDER_1,new_wall_coords[0],new_wall_coords[1]] == 1 or self.grid[HIDER_2,new_wall_coords[0],new_wall_coords[1]] == 1:
                            isValidMove = False
                        else:
                            isValidMove = True
                            movable_wall_moved = True
                            #move the movable wall
                            self.grid[MOVABLE_WALL,potential_new_coords[0],potential_new_coords[1]] = 0
                            self.grid[MOVABLE_WALL,new_wall_coords[0],new_wall_coords[1]] = 1
                else:
                    isValidMove = True

            if isValidMove:
                #move is valid
                #if there is a movable wall, move it if it can be
                self.infos[agent]["coords"] = potential_new_coords
            else:
                self.infos[agent]["coords"] = coord_be4
        return self.write_grid_from_info(self.grid,self.infos)

    def step(self,actions):

        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        rewards = {}

        #print(f'called step with actions {actions}')

        new_state = self.apply_actions(actions)
        #updates grid and coords
        self.grid = new_state

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        observations = {agent: new_state for agent in self.agents}
        self.state = observations


        #update observations dict
        #update self.state

        #COMPUTE REWARD TODO
        rewards = self.compute_rewards_all_agents()

        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

ALL_PYGAME_FRAMES = None

