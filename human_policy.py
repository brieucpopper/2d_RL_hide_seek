
import random
from env import PettingZooGridWorld
import matplotlib.pyplot as plt
import os
from matplotlib import animation

def l1_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class HumanPolicy:
    def __init__(self, env):
        self.env = env

    def get_action(self,agent):
        
        if 'pred' in agent:
            loc_pred = self.env.agent_positions[agent]
            loc_pray = {k : v for k,v in self.env.agent_positions.items() if 'prey' in k}
            #Get closest prey, using L1 distance
            closest_prey = min(loc_pray, key=lambda x: l1_distance(loc_pred, loc_pray[x]))
            distance = l1_distance(loc_pred, loc_pray[closest_prey])
            if distance == 1:
                return 4
            else:
                
                
            
                
            #Move towards closest prey
                pred_x, pred_y = loc_pred
                prey_x, prey_y = loc_pray[closest_prey]
                
                if pred_x < prey_x and pred_y < prey_y:
                    #return 0 or 2 with equal probability
                    return random.choice([0,2])
                if pred_x < prey_x and pred_y > prey_y:
                    return random.choice([0,3])
                if pred_x > prey_x and pred_y < prey_y:
                    return random.choice([1,2])
                if pred_x > prey_x and pred_y > prey_y:
                    return random.choice([1,3])
                
                if pred_x == prey_x and pred_y < prey_y:
                    return 2
                if pred_x == prey_x and pred_y > prey_y:
                    return 3
                if pred_x < prey_x and pred_y == prey_y:
                    return 0
                if pred_x > prey_x and pred_y == prey_y:
                    return 1
                
                
                    
        else:
            return random.choice([4,0,4,1,4,2,4,3])
                
            
        
        
            #Special cases to pay attention to : Being on an edge, wall ?, being just next to prey, being in a line


def demo_policy(max_steps, grid_size):
    
    env = PettingZooGridWorld(grid_size)
    env.reset()
    
    cum_reward = None
    frames = []
    for _ in range(max_steps):
        agent = env.agent_selector.next()
        policy = HumanPolicy(env)
        action = policy.get_action(agent)
        
        observations, rewards, dones, infos = env.step({agent: action})
        
        if cum_reward is None:
            cum_reward = rewards
        else:
            for agent in env.agents:
                cum_reward[agent] += rewards[agent]
        frames.append(env.render(mode='rgb_array'))
    return frames, cum_reward


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
