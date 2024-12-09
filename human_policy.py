
import random
from env_clean import PettingZooGridWorld
import matplotlib.pyplot as plt
import os
from matplotlib import animation
from buffer import ReplayBuffer
import torch
from agent import CQLSAC
def l1_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class HumanPolicy:
    def __init__(self, env,randomness_prey = 0.1,randomness_pred = 0.1):
        self.env = env
        self.randomness_prey = randomness_prey
        self.randomness_pred = randomness_pred

    def get_action(self):
        
        agent_list = self.env.agent_positions.keys()
        action = {}
        for agent in agent_list:
        
            if 'pred' in agent:
                
                
                if random.random() < self.randomness_pred:
                    action[agent] = random.choice([0,1,2,3,4])
                    continue
                else:
                
                    ind_pred = agent[5]
                    loc_pred = self.env.agent_positions[agent]
                    #loc_pray = {k : v for k,v in self.env.agent_positions.items() if 'prey' in k}
                    #Get closest prey, using L1 distance
                    loc_pray = self.env.agent_positions[f'prey_{ind_pred}'] #{ind_pred}
                    #closest_prey = min(loc_pray, key=lambda x: l1_distance(loc_pred, loc_pray[x]))
                    #distance = l1_distance(loc_pred, loc_pray)
                
                    
                    
                
                    
                
                    pred_x, pred_y = loc_pred
                    prey_x, prey_y = loc_pray#[closest_prey]
                    
                    can_go_up = self.env.grid[0,pred_y,pred_x-1] == 0
                    can_go_down = self.env.grid[0,pred_y,pred_x+1] == 0
                    can_go_left = self.env.grid[0,pred_y-1,pred_x] == 0
                    can_go_right = self.env.grid[0,pred_y+1,pred_x] == 0
                    
                    action_available = {0:can_go_down,1:can_go_up,2:can_go_right,3:can_go_left}
                    
                    if pred_x == prey_x and pred_y == prey_y:
                        action[agent] = 4
                        continue
                    
                    elif pred_x < prey_x and pred_y < prey_y:
                        #return 0 or 2 with equal probability
                        possible_actions = [0,2]
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [4,0,3]
                        action[agent] = random.choice(possible_actions)
                        #return random.choice([0,2])
                    elif pred_x < prey_x and pred_y > prey_y:
                        possible_actions = [0,3]
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [4,1,2]
                        action[agent] = random.choice(possible_actions)
                        
                        
                        
                        #return random.choice([0,3])
                    elif pred_x > prey_x and pred_y < prey_y:
                        
                        possible_actions = [1,2]
                        
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [4,0,3]
                        action[agent] = random.choice(possible_actions)
                        
                        
                        #return random.choice([1,2])
                    elif pred_x > prey_x and pred_y > prey_y:
                        #return random.choice([1,3])
                        
                        possible_actions = [1,3] ### GOOD
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [4,0,2]
                        action[agent] = random.choice(possible_actions)
                    
                    elif pred_x == prey_x and pred_y < prey_y:
                        #return 2 : go right
                        if can_go_right:
                            action[agent] = 2
                        else:
                            action[agent] = random.choice([0,1])
                        
                    elif pred_x == prey_x and pred_y > prey_y:
                        #return 3 : go left
                        if can_go_up:
                            action[agent] = 3
                        else:
                            action[agent] = random.choice([0,1])
                        
                    elif pred_x < prey_x and pred_y == prey_y:
                        #return 0 : go down
                        if can_go_down:
                            action[agent] = 0
                        else:
                            action[agent] = random.choice([3,2])
                        
                    elif pred_x > prey_x and pred_y == prey_y:
                        #return 1 : go up
                        if can_go_up:
                            action[agent] = 1
                        else:
                            action[agent] = random.choice([3,2])
                    else:
                        action[agent] = random.choice([0,1,2,3,4])
                    
                        
                            
            else:
                #return random.choice([4,0,4,1,4,2,4,3])
                #action[agent] = random.choice([4,0,4,1,4,2,4,3])
                
                if random.random() < self.randomness_prey:
                    action[agent] = random.choice([0,1,2,3])
                    continue
                else:
                
                    
                    ind_pray = agent[5]
                    loc_pray = self.env.agent_positions[agent]
                    loc_pred = self.env.agent_positions[f'pred_{ind_pray}'] #ind_pray
                    pred_x, pred_y = loc_pred
                    prey_x, prey_y = loc_pray#
                    
                    can_go_up = self.env.grid[0,prey_y,prey_x-1] == 0
                    can_go_down = self.env.grid[0,prey_y,prey_x+1] == 0
                    can_go_left = self.env.grid[0,prey_y-1,prey_x] == 0
                    can_go_right = self.env.grid[0,prey_y+1,prey_x] == 0
                    
                    action_available = {0:can_go_down,1:can_go_up,2:can_go_right,3:can_go_left}
                    
                    if pred_x == prey_x and pred_y == prey_y:
                        possible_actions = [0,1,2,3]
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [4]
                        action[agent] = random.choice(possible_actions)
                    elif pred_x < prey_x and pred_y < prey_y:
                        #return 0 or 2 with equal probability
                        possible_actions = [0,2]
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [3,1]
                        action[agent] = random.choice(possible_actions)
                        #return random.choice([0,2])
                    elif pred_x < prey_x and pred_y > prey_y:
                        possible_actions = [3,0]
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [1,2]
                        action[agent] = random.choice(possible_actions)
                        
                        
                        
                        #return random.choice([0,3])
                    elif pred_x > prey_x and pred_y < prey_y:
                        
                        possible_actions = [1,2]
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [0,3]
                        action[agent] = random.choice(possible_actions)
                        
                        
                        #return random.choice([1,2])
                    elif pred_x > prey_x and pred_y > prey_y:
                        #return random.choice([1,3])
                        
                        possible_actions = [1,3]
                        for act,available in action_available.items():
                            if not available and act in possible_actions:
                                possible_actions.remove(act)
                        if len(possible_actions) == 0:
                            possible_actions = [0,2]
                        action[agent] = random.choice(possible_actions)
                    elif pred_x == prey_x and pred_y < prey_y:
                        #return 2 : go right
                        #{0:can_go_down,1:can_go_up,2:can_go_right,3:can_go_left}
                        if can_go_right:
                            action[agent] = 2
                        else:
                            action[agent] = random.choice([0,1])
                        
                    elif pred_x == prey_x and pred_y > prey_y:
                        #return 3 : go left
                        if can_go_left:
                            action[agent] = 3
                        else:
                            action[agent] = random.choice([0,1])
                        
                    elif pred_x < prey_x and pred_y == prey_y:
                        #return 0 : go down
                        if can_go_down:
                            action[agent] = 0
                        else:
                            action[agent] = random.choice([3,2])
                        
                    elif pred_x > prey_x and pred_y == prey_y:
                        #return 1 : go up
                        if can_go_up:
                            action[agent] = 1
                        else:
                            action[agent] = random.choice([3,2])
                    
                    else:
                        action[agent] = random.choice([0,1,2,3,4])    
                    
        #for agent in agent_list:
        #    if 'pred' in agent:
        #        action[agent] =4 
        
        return action
                #Introduce randomness in the policy?

def demo_policy(grid_size, max_steps=1000,randomness_pred = 0.1,randomness_prey = 0.1,n_walls = 0):
    
    
    
    env = PettingZooGridWorld(grid_size=grid_size,walls = True, num_walls = n_walls)
    env.reset()
    cum_reward = None
    frames = []
    print(env.agent_positions)
    for _ in range(max_steps):
        
        policy = HumanPolicy(env,randomness_prey=  randomness_prey,randomness_pred = randomness_pred)
        action = policy.get_action()
        
        observations, rewards, dones, infos = env.step(action)
        
        if cum_reward is None:
            cum_reward = rewards
        else:
            for agent in env.agents:
                cum_reward[agent] += rewards[agent]
        frames.append(env.render(mode='rgb_array'))
    #print(len(frames))
    return frames, cum_reward

def demo_policy_CQL(env_size, n_walls,agentCQL,max_steps=1000):
    
    
    #agentCQL = CQLSAC(state_size=3,action_size=1,device='mps',env_size=env_size)
    #agentCQL.actor_local.load_state_dict(torch.load(path))
    env = PettingZooGridWorld(env_size,walls = True, num_walls = n_walls)
    env.reset()
    cum_reward = None
    frames = []
    for _ in range(max_steps):
        actions = {}
        for agent in env.agents:
            state = env.get_observations()[agent]
            action = agentCQL.actor_local.get_det_action((torch.tensor(state,dtype=torch.float32).unsqueeze(0).to('mps'))).item()
            actions[agent] = action
        observations, rewards, dones, infos = env.step(actions)
        if cum_reward is None:
            cum_reward = rewards
        else:
            for agent in env.agents:
                cum_reward[agent] += rewards[agent]
        frames.append(env.render(mode='rgb_array'))
    return frames, cum_reward
    
def demo_policy_CQL_distinct(env_size, n_walls,agentCQL_pred,agentCQL_prey,max_steps=1000):
    
    
    #agentCQL = CQLSAC(state_size=3,action_size=1,device='mps',env_size=env_size)
    #agentCQL.actor_local.load_state_dict(torch.load(path))
    env = PettingZooGridWorld(env_size,walls = True, num_walls = n_walls)
    env.reset()
    cum_reward = None
    frames = []
    for _ in range(max_steps):
        actions = {}
        for agent in env.agents:
            state = env.get_observations()[agent]
            if 'prey' in agent:
                action = agentCQL_prey.actor_local.get_det_action((torch.tensor(state,dtype=torch.float32).unsqueeze(0).to('mps'))).item()
            else:
                action = agentCQL_pred.actor_local.get_det_action((torch.tensor(state,dtype=torch.float32).unsqueeze(0).to('mps'))).item()
            actions[agent] = action
        observations, rewards, dones, infos = env.step(actions)
        if cum_reward is None:
            cum_reward = rewards
        else:
            for agent in env.agents:
                cum_reward[agent] += rewards[agent]
        frames.append(env.render(mode='rgb_array'))
    return frames, cum_reward
    

def create_buffer(env_size,n_walls,batch_size = 256,device = 'mps',buffer_size = 1000000,randomness_prey= 0.05,randomness_pred = 0.05,agentType = None,mixed = False):
    
    buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
    while len(buffer) < buffer_size:
        k = 0
        env = PettingZooGridWorld(env_size,walls = True, num_walls = n_walls)
        for i in range(100):
            k+=1
            observations = env.get_observations()
            if mixed and k % 2 == 0:
                policy = HumanPolicy(env,randomness_pred= 0.9,randomness_prey = 0.9)
            else:
                policy = HumanPolicy(env,randomness_pred= randomness_pred,randomness_prey = randomness_prey)
            
            action = policy.get_action()
            next_observations, rewards, dones, infos = env.step(action)
            
            for agent in env.agents:
                if agentType is None:
                    buffer.add(observations[agent], action[agent], rewards[agent], next_observations[agent], dones[agent])
                else:
                    if agentType in agent:
                        buffer.add(observations[agent], action[agent], rewards[agent], next_observations[agent], dones[agent])
                        
                        
                
    return buffer

def save_frames_as_gif(frames,method_name = 'human'):
    
    ARTIFACT_DIRECTORY = 'artifacts'
    if not os.path.exists(ARTIFACT_DIRECTORY):
        os.makedirs(ARTIFACT_DIRECTORY)
    path = f'{ARTIFACT_DIRECTORY}/{method_name}_policy.gif'
    
    # controls frame size
    fig = plt.figure(
        figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0),
        dpi=72
    )

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames = len(frames), interval=1
    )
    anim.save(
        path,
        writer='imagemagick', fps=60
    )
    plt.close(fig)
    return path
