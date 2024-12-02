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

GRID_SIZE = 8
CKPT_PATH = "/home/bpopper/letsgo/2d_RL_hide_seek/PARALLEL/weights/best_model_pred1.pth"

RANDOM = 3
TRAINING = 2

NUM_THINGS = 6 # number of things in the grid wall, pred1, pred2, h1, h2, movablewall

POLICIES = [
    TRAINING, # pred_1
    RANDOM,    # pred_2
    RANDOM,  # hider_1
    RANDOM]    # hider_2
# This should be either TRAINING, RANDOM, or a string that is a path to a ckpt for each agent




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
        hidden = self.network(x / 1.0)  # Normalize input to [0, 1]
        
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
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


    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.4
    clip_coef = 0.08
    gamma = 0.95
    batch_size = 64
    max_cycles = 250
    total_episodes = 1500

    do_train = True

    """ ENV SETUP """
    env = movable_wall_parallel.parallel_env(grid_size=GRID_SIZE,walls=False)

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    agent_pred_1 = Agent(num_actions=num_actions).to(device)
    optimizer_p1 = optim.Adam(agent_pred_1.parameters(), lr=0.0001, eps=1e-5)
    agent_pred_2 = Agent(num_actions=num_actions).to(device)
    optimizer_p2 = optim.Adam(agent_pred_2.parameters(), lr=0.0001, eps=1e-5)
    agent_hider_1 = Agent(num_actions=num_actions).to(device)
    optimizer_h1 = optim.Adam(agent_hider_1.parameters(), lr=0.0001, eps=1e-5)
    agent_hider_2 = Agent(num_actions=num_actions).to(device)
    optimizer_h2 = optim.Adam(agent_hider_2.parameters(), lr=0.0001, eps=1e-5)

    def get_agent_from_idx(idx):
        if idx == 0:
            return agent_pred_1
        elif idx == 1:
            return agent_pred_2
        elif idx == 2:
            return agent_hider_1
        elif idx == 3:
            return agent_hider_2
    
    def get_agent_str_from_idx(idx):
        if idx == 0:
            return "pred_1"
        elif idx == 1:
            return "pred_2"
        elif idx == 2:
            return "hider_1"
        elif idx == 3:
            return "hider_2"
    
    def get_agent_idx_from_key(key):
        if key == "pred_1":
            return 0
        elif key == "pred_2":
            return 1
        elif key == "hider_1":
            return 2
        elif key == "hider_2":
            return 3

    # load pretrained agents
    for i, policy in enumerate(POLICIES):
        #if is a string
        if isinstance(policy, str):
            if i == 0:
                agent_pred_1.load_state_dict(torch.load(policy))
                #freeze params
                for param in agent_pred_1.parameters():
                    param.requires_grad = False
            elif i == 1:
                agent_pred_2.load_state_dict(torch.load(policy))
                #freeze params
                for param in agent_pred_2.parameters():
                    param.requires_grad = False
            elif i == 2:
                agent_hider_1.load_state_dict(torch.load(policy))
                #freeze params
                for param in agent_hider_1.parameters():
                    param.requires_grad = False
            elif i == 3:
                agent_hider_2.load_state_dict(torch.load(policy))
                #freeze params
                for param in agent_hider_2.parameters():
                    param.requires_grad = False


    

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, NUM_THINGS,GRID_SIZE,GRID_SIZE)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)


    if do_train:
        """ TRAINING LOGIC """
        
        # train for n number of episodes
        best_smoothed_return = -10000
        all_returnspred1 = []
        all_returnspred2 = []
        all_returnshider1 = []
        all_returnshider2 = []
        best_pred1 = - 10000
        for episode in range(total_episodes):
            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs, info = env.reset(seed=None)
                # reset the episodic return
                total_episodic_return = 0

                # each episode has num_steps
                for step in range(0, max_cycles):
                    # rollover the observation
                    
                    obs = next_obs.copy()
                    # get action for first agent from the trained agent
                    # get random actions for other agents
                    actions = torch.zeros(num_agents, dtype=torch.long).to(device)
                    logprobs = torch.zeros(num_agents).to(device)
                    values = torch.zeros(num_agents).to(device)

                    for i in range(0, num_agents):
                        if POLICIES[i] == TRAINING:
                            actions[i], logprobs[i], _, values[i] = get_agent_from_idx(i).get_action_and_value(torch.tensor(obs[get_agent_str_from_idx(i)]).unsqueeze(0).to(device))
                        elif POLICIES[i] == RANDOM:
                            actions[i] = torch.randint(0, num_actions, (1,)).to(device)
                            logprobs[i] = torch.log(torch.tensor(1.0/num_actions))
                            values[i] = 0.0

            
                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = env.step(
                        unbatchify(actions, env)
                    )

                    # add to episode storage
                    for key in ["pred_1", "pred_2", "hider_1", "hider_2"]:

                        rb_obs[step,get_agent_idx_from_key(key)] = torch.tensor(obs[key]).unsqueeze(0).to(device)

                    rb_rewards[step] = batchify(rewards, device)
                    rb_terms[step] = batchify(terms, device)
                    rb_actions[step] = actions
                    rb_logprobs[step] = logprobs
                    rb_values[step] = values

                    # compute episodic return
                    total_episodic_return += rb_rewards[step].cpu().numpy()

                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                        end_step = step
                        break

        
            with torch.no_grad():
    
                rb_advantages = torch.zeros_like(rb_rewards).to(device)
                for policy_idx in range(num_agents):
                    if POLICIES[policy_idx] == TRAINING:
                        
                        for t in reversed(range(end_step)):
                            delta = (
                                rb_rewards[t, policy_idx]  
                                + gamma * rb_values[t + 1, policy_idx] * rb_terms[t + 1, policy_idx]
                                - rb_values[t, policy_idx]
                            )
                            rb_advantages[t, policy_idx] = delta + gamma * gamma * rb_advantages[t + 1, policy_idx]

                rb_returns = rb_advantages + rb_values

            in_training = [POLICIES[i] == TRAINING for i in range(num_agents)]

            ordered_training_policies = [i for i in range(num_agents) if POLICIES[i] == TRAINING] 
            #for example if pred_1 and hider_1 are training, ordered_training_policies = [0,2]

            #mapping : takes the original policy idx and returns where it is in the ordered_training_policies
            # for example if pred_1 and hider_1 are training, mapping(0) = 0, mapping(2) = 1

            def mapping(original_idx):
                return ordered_training_policies.index(original_idx)

            # convert our episodes to batch of individual transitions (only for the trained agents)
            b_obs = rb_obs[:end_step, in_training]
            b_logprobs = rb_logprobs[:end_step, in_training]
            b_actions = rb_actions[:end_step, in_training]
            b_returns = rb_returns[:end_step, in_training]
            b_values = rb_values[:end_step, in_training]
            b_advantages = rb_advantages[:end_step, in_training]

            # Optimizing the policy and value network
            b_index = np.arange(len(b_obs))
            clip_fracs = []
            for repeat in range(3):
                # shuffle the indices we use to access the data
                np.random.shuffle(b_index)
                for start in range(0, len(b_obs), batch_size):
                    # select the indices we want to train on
                    end = start + batch_size
                    batch_index = b_index[start:end]
                    for policy_idx in range(num_agents):
                        if POLICIES[policy_idx] == TRAINING:

                            #print(f'shape of actions {b_actions.long()[batch_index][:,1].shape}')
                            
                            _, newlogprob, entropy, value = get_agent_from_idx(policy_idx).get_action_and_value(
                            b_obs[batch_index][:,mapping(policy_idx),:,:,:], b_actions.long()[batch_index][:,mapping(policy_idx)]
                            )
                            
                            logratio = newlogprob - b_logprobs[batch_index][:,mapping(policy_idx)]
                            ratio = logratio.exp()

                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clip_fracs += [
                                    ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                                ]

                            # normalize advantages
                            advantages = b_advantages[batch_index]
                            advantages = (advantages - advantages.mean()) / (
                                advantages.std() + 1e-8
                            )

                            # Policy loss
                            pg_loss1 = -b_advantages[batch_index][:,mapping(policy_idx)] * ratio
                            pg_loss2 = -b_advantages[batch_index][:,mapping(policy_idx)] * torch.clamp(
                                ratio, 1 - clip_coef, 1 + clip_coef
                            )
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            value = value.flatten()
                            v_loss_unclipped = (value - b_returns[batch_index][:,mapping(policy_idx)]) ** 2
                            v_clipped = b_values[batch_index][:,mapping(policy_idx)] + torch.clamp(
                                value - b_values[batch_index][:,mapping(policy_idx)],
                                -clip_coef,
                                clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[batch_index][:,mapping(policy_idx)]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()

                            entropy_loss = entropy.mean()
                            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                            def get_optimizer_from_idx(idx):
                                if idx == 0:
                                    return optimizer_p1
                                elif idx == 1:
                                    return optimizer_p2
                                elif idx == 2:
                                    return optimizer_h1
                                elif idx == 3:
                                    return optimizer_h2
                            
                            optimizer = get_optimizer_from_idx(policy_idx)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            print(f"Training episode {episode}")
            print(f"Episodic Return: {(total_episodic_return)}")
            print(f" YOU CAN ADD CODE TO SAVE CHECKPOINTS HERE")
            print(f" make sure the rewards are defined well in parallel.py")
            all_returnspred1.append(total_episodic_return[0])
            all_returnspred2.append(total_episodic_return[1])
            all_returnshider1.append(total_episodic_return[2])
            all_returnshider2.append(total_episodic_return[3])

            #print smoothed returns average last 20

            print(f"Smoothed Returns for pred_1: {np.mean(all_returnspred1[-20:])}")
            print(f"Smoothed Returns for pred_2: {np.mean(all_returnspred2[-20:])}")
            print(f"Smoothed Returns for hider_1: {np.mean(all_returnshider1[-20:])}")
            print(f"Smoothed Returns for hider_2: {np.mean(all_returnshider2[-20:])}")

            if best_pred1 < np.mean(all_returnspred1[-20:]):
                best_pred1 = np.mean(all_returnspred1[-20:])
                torch.save(agent_pred_1.state_dict(), "./best_pred1_nowalls_6x8x8.pth")
                print("Saved best model for pred_1")

            print(f"Episode Length: {end_step}")
            print("")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            print("\n-------------------------------------------\n")
