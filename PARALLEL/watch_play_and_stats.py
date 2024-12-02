import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


import movable_wall_parallel

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

# policies = [
#     '/home/bpopper/letsgo/2d_RL_hide_seek/PARALLEL/weights/best_model.pth',
#     None,
#     '/home/bpopper/letsgo/2d_RL_hide_seek/PARALLEL/best_model_fleee.pth',
#     None]

#both trained :  approx reward 250 for hider

# policies = [
#     None,
#     None,
#     '/home/bpopper/letsgo/2d_RL_hide_seek/PARALLEL/weights/best_model_hider1.pth',
#     None]
# #ith only trained hider : approx rewared 1800 for hider


policies = [
    '/home/bpopper/letsgo/2d_RL_hide_seek/PARALLEL/weights/best_pred1_movablewall.pth',
    None,
    None,
    None]
#with trained seeker : approx 155 reward for hider

#either None : random ; or a path

GRID_SIZE = 8
NUM_THINGS = 6

env = movable_wall_parallel.parallel_env(grid_size=GRID_SIZE,render_mode="human",walls=False)

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        # CNN architecture inspired by DQN for Atari
        self.network = nn.Sequential(
            nn.Conv2d(NUM_THINGS, 32, kernel_size=3, stride=1, padding=1),  # Output: 32 x 7 x 7
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 7 x 7
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 7 x 7
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),  # Output: 64 * 7 * 7 = 3136
        )
        self.actor = self._layer_init(nn.Linear(64*GRID_SIZE**2, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(64*GRID_SIZE**2, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 1.0))  # Normalize input to [0, 1]

    def get_action_and_value(self, x, action=None):
        #print(x)
        #print(x.shape)
        hidden = self.network(x / 1.0)  # Normalize input to [0, 1]
        
        logits = self.actor(hidden)
        #print(f' in get_action_and_valuelogits: {logits}')
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        #print(f' in get_action_and_value: {action}')
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    

def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


if __name__ == "__main__":




    """ ENV SETUP """
    

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ RENDER THE POLICY """
    

    
    agent_pred1 = Agent(num_actions=num_actions).to(device)
    if policies[0] is not None:
        agent_pred1.load_state_dict(torch.load(policies[0]))

    agent_pred2 = Agent(num_actions=num_actions).to(device)
    if policies[1] is not None:
        agent_pred2.load_state_dict(torch.load(policies[1]))

    agent_flee1 = Agent(num_actions=num_actions).to(device)
    if policies[2] is not None:
        agent_flee1.load_state_dict(torch.load(policies[2]))

    agent_flee2 = Agent(num_actions=num_actions).to(device)
    if policies[3] is not None:
        agent_flee2.load_state_dict(torch.load(policies[3]))




    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            #obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            total_ep_rew = {'pred_1':0, 'pred_2':0, 'hider_1':0, 'hider_2':0}
            while not any(terms) and not any(truncs):

                action_p1, logprob_p1, _, value_p1 = agent_pred1.get_action_and_value(torch.tensor(obs['pred_1']).unsqueeze(0).to(device))
                action_p2, logprob_p2, _, value_p2 = agent_pred2.get_action_and_value(torch.tensor(obs['pred_2']).unsqueeze(0).to(device))
                action_h1, logprob_h1, _, value_h1 = agent_flee1.get_action_and_value(torch.tensor(obs['hider_1']).unsqueeze(0).to(device))
                action_h2, logprob_h2, _, value_h2 = agent_flee2.get_action_and_value(torch.tensor(obs['hider_2']).unsqueeze(0).to(device))

                actions = torch.cat([action_p1, action_p2, action_h1, action_h2])
                #print(actions)  
                
                for idx,p in enumerate(policies):
                    if p == None:
                        actions[idx] = torch.randint(0, num_actions, (1,)).to(device)

                #print(actions)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                #obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]

                total_ep_rew['pred_1'] += rewards['pred_1']
                total_ep_rew['pred_2'] += rewards['pred_2']
                total_ep_rew['hider_1'] += rewards['hider_1']
                total_ep_rew['hider_2'] += rewards['hider_2']


            print(f"Episode {episode} rewards: {total_ep_rew}")