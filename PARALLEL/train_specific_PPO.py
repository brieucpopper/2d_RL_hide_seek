import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import parallel
import torch
import movable_wall_parallel
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import wandb
RANDOM = 42



############################ HIGHLY IMPORTANT VARIABLES TO SET######################################
GRID_SIZE = 7
NUM_THINGS = 6 # number of things in the grid wall, pred1, pred2, h1, h2, movablewall


INITIALIZATIONS = [
    RANDOM, # pred_1
    RANDOM,    # pred_2
    RANDOM,     # hider_1
    RANDOM,
    ]    # hider_2
#should be either RANDOM ; or a path to a pretrained checkpoint (a String)



IS_TRAINING =   [
    False,
    False,
    True,
    False
]
#either True or False, if False, weights are frozen (or if random it will stay random)


envname = 'mparallel-walls' #just for wandb logging
CUSTOMENV = movable_wall_parallel.parallel_env(grid_size=GRID_SIZE,walls=True)
# change architecture if needed

ent_coef = 0.4
vf_coef = 0.2
clip_coef = 0.1
gamma = 0.975
batch_size = 64
max_cycles = 200
total_episodes = 1600
PPO_STEPS = 3

reminder = '''DONT FORGET TO ADD CODE TO SAVE CHECKPOINTS IF YOU WANT TO DO THAT'''
print(reminder)

##################################################################################################




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Agent2(nn.Module):
    def __init__(self, num_actions, num_input_channels=NUM_THINGS):
        super().__init__()
        
        # Deeper CNN with residual connections and batch normalization
        self.network = nn.Sequential(
            # Initial feature extraction block
            nn.Conv2d(num_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Residual block 1
            self._make_residual_block(32, 64),
            
            # Residual block 2
            self._make_residual_block(64, 128),
            
            # Additional conv layer for more feature extraction
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Flatten()
        )
        
        # Compute the flattened feature size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_input_channels, 7, 7)
            feature_size = self.network(dummy_input).shape[1]
        
        # Improved actor and critic heads with layer normalization
        self.actor_hidden = self._layer_init(nn.Linear(feature_size, 256))
        self.actor_norm = nn.LayerNorm(256)
        self.actor = self._layer_init(nn.Linear(256, num_actions), std=0.01)
        
        self.critic_hidden = self._layer_init(nn.Linear(feature_size, 256))
        self.critic_norm = nn.LayerNorm(256)
        self.critic = self._layer_init(nn.Linear(256, 1))
    
    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block with batch normalization."""
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        return nn.Sequential(
            block,
            nn.ReLU(inplace=True)
        )
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize layer weights using orthogonal initialization."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_value(self, x):
        """Get the critic's value estimate."""
        hidden = self.network(x / 255.0)  # Normalize input to [0, 1]
        hidden = F.relu(self.critic_norm(self.critic_hidden(hidden)))
        return self.critic(hidden)
    
    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value."""
        hidden = self.network(x / 255.0)  # Normalize input to [0, 1]
        
        # Actor head with separate hidden layer and normalization
        actor_hidden = F.relu(self.actor_norm(self.actor_hidden(hidden)))
        logits = self.actor(actor_hidden)
        
        # Use softmax to ensure proper probability distribution
        probs = Categorical(probs=F.softmax(logits, dim=-1))
        
        if action is None:
            action = probs.sample()
        
        return (
            action, 
            probs.log_prob(action), 
            probs.entropy(), 
            self.get_value(x)
        )

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        # CNN architecture inspired by DQN for Atari
        self.network = nn.Sequential(
            nn.Conv2d(NUM_THINGS, 32, kernel_size=3, stride=1, padding=1),  # Output: 32 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),  # Output: 64 * 7 * 7 = 3136
        )
        self.actor = self._layer_init(nn.Linear(3136, num_actions), std=0.01) #TODO depends on GRID_SIZE
        self.critic = self._layer_init(nn.Linear(3136, 1))

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
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = torch.tensor(obs).to(device)
    return obs

def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)
    return x

def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x

if __name__ == "__main__":
        #init wandb with all the above params
    wandb.init(
            project="multi-agent-ppo",  # Set your project name
            config={
                "env": envname,
                "GRID_SIZE": GRID_SIZE,
                "NUM_THINGS": NUM_THINGS,
                "INITIALIZATIONS": INITIALIZATIONS,
                "IS_TRAINING": IS_TRAINING,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "clip_coef": clip_coef,
                "gamma": gamma,
                "batch_size": batch_size,
                "max_cycles": max_cycles,
                "total_episodes": total_episodes,
                "PPO_STEPS": PPO_STEPS,
            }
    )
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    """ ENV SETUP """
    env = CUSTOMENV

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

   

    """ LEARNER SETUP """
    # Create a list of agents, one for each training agent
    training_agents = []
    optimizers = []
    training_agent_indices = [i for i, training in enumerate(IS_TRAINING) if training]
    frozen_agent_indices = [i for i, training in enumerate(IS_TRAINING) if not training and INITIALIZATIONS[i] != RANDOM]
    # a frozen is one that is NOT TRAINING and NOT RANDOM
    for idx in training_agent_indices:

        
        agent = Agent(num_actions=num_actions).to(device)

        if INITIALIZATIONS[idx] != RANDOM:
            agent.load_state_dict(torch.load(INITIALIZATIONS[idx]))
            print(f'loaded from {INITIALIZATIONS[idx]}')

        optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)
        training_agents.append(agent)
        optimizers.append(optimizer)
    
    frozen_agents = [] # These agents are not random, but are NOT TRAINING ; initialized with a checkpoint
    for idx, init in enumerate(INITIALIZATIONS):
        if init != RANDOM and not IS_TRAINING[idx]:
            agent = Agent(num_actions=num_actions).to(device)
            agent.load_state_dict(torch.load(init))
            agent.eval()
            #freeze weights
            for param in agent.parameters():
                param.requires_grad = False
            print(f' loaded from {init}')
            frozen_agents.append(agent)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, NUM_THINGS,GRID_SIZE,GRID_SIZE)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOGIC """
    # Track returns for all agents
    all_returns = [[] for _ in range(num_agents)]

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
                obs = batchify_obs(next_obs, device)

                # get action for first agent from the trained agents
                # get random actions for other agents
                actions = torch.zeros(num_agents, dtype=torch.long).to(device)
                logprobs = torch.zeros(num_agents).to(device)
                values = torch.zeros(num_agents).to(device)

                # Process each agent
                for i in range(num_agents):
                    if IS_TRAINING[i]:
                        # Find the index of this training agent among training agents
                        train_idx = training_agent_indices.index(i)
                        # Get action and value for training agent
                        agent_obs = obs[i].unsqueeze(0)
                        actions[i], logprobs[i], _, values[i] = training_agents[train_idx].get_action_and_value(agent_obs)
                    elif INITIALIZATIONS[i] != RANDOM:
                        #this is a frozen agent (not training, but not random because it has a checkpoint)
                        frozen_idx = frozen_agent_indices.index(i)
                        agent_obs = obs[i].unsqueeze(0)
                        actions[i], logprobs[i], _, values[i] = frozen_agents[frozen_idx].get_action_and_value(agent_obs)

                        logprobs[i] = torch.log(torch.tensor(1.0/num_actions))
                        values[i] = 0.0  # No value estimation for frozen agents
                    else:
                        # Random action for random agents
                        actions[i] = torch.randint(0, num_actions, (1,)).to(device)
                        logprobs[i] = torch.log(torch.tensor(1.0/num_actions))
                        values[i] = 0.0  # No value estimation for random agents

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
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

        # Train only the specified agents
        for train_idx, agent_idx in enumerate(training_agent_indices):
            # Bootstrap value and advantages only for the training agent
            with torch.no_grad():
                rb_advantages = torch.zeros_like(rb_rewards).to(device)
                for t in reversed(range(end_step)):
                    delta = (
                        rb_rewards[t, agent_idx]  # Only specific agent's reward
                        + gamma * rb_values[t + 1, agent_idx] * rb_terms[t + 1, agent_idx]
                        - rb_values[t, agent_idx]
                    )
                    rb_advantages[t, agent_idx] = delta + gamma * gamma * rb_advantages[t + 1, agent_idx]
                rb_returns = rb_advantages + rb_values

            # convert our episodes to batch of individual transitions (only for specific agent)
            b_obs = rb_obs[:end_step, agent_idx]
            b_logprobs = rb_logprobs[:end_step, agent_idx]
            b_actions = rb_actions[:end_step, agent_idx]
            b_returns = rb_returns[:end_step, agent_idx]
            b_values = rb_values[:end_step, agent_idx]
            b_advantages = rb_advantages[:end_step, agent_idx]

            # Optimizing the policy and value network
            b_index = np.arange(len(b_obs))
            clip_fracs = []
            for repeat in range(PPO_STEPS):
                # shuffle the indices we use to access the data
                np.random.shuffle(b_index)
                for start in range(0, len(b_obs), batch_size):
                    # select the indices we want to train on
                    end = start + batch_size
                    batch_index = b_index[start:end]

                    _, newlogprob, entropy, value = training_agents[train_idx].get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index]
                    )
                    logratio = newlogprob - b_logprobs[batch_index]
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
                    pg_loss1 = -b_advantages[batch_index] * ratio
                    pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                    optimizers[train_idx].zero_grad()
                    loss.backward()
                    optimizers[train_idx].step()

            # Store returns for the training agents
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Accumulate returns for all agents
        for i in range(num_agents):
            all_returns[i].append(total_episodic_return[i])

        if episode % 10 == 0:
            print(f"Training episode {episode}")
            print(f"Episodic Return: {(total_episodic_return)}")

            # Print smoothed returns for each agent
            for i in range(num_agents):
                status = None

                if IS_TRAINING[i]:
                    status = "Training"
                elif INITIALIZATIONS[i] == RANDOM:
                    status = "Random"
                else:
                    status = "Frozen"


                print(f"Smoothed Returns for agent_{i} ({status}): {np.mean(all_returns[i][-20:])}")

            print(f"Episode Length: {end_step}")
            print("")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            print("\n-------------------------------------------\n")

        #log all with wandb
        wandb.log({
            "Ep return pred1": total_episodic_return[0],
            "Ep return pred2": total_episodic_return[1],
            "Ep return hider1": total_episodic_return[2],
            "Ep return hider2": total_episodic_return[3],
            "Episode Length": end_step,
            "Value Loss": v_loss.item(),
            "Policy Loss": pg_loss.item(),
            "Old Approx KL": old_approx_kl.item(),
            "Approx KL": approx_kl.item(),
            "Clip Fraction": np.mean(clip_fracs),
            "Explained Variance": explained_var.item()
        })

        #if for pred_1 (index 0) episode return and smoothed are greater than -200, save the model

        # if total_episodic_return[0] > -210 and np.mean(all_returns[0][-20:]) > -210:
        #     #create dir
        #     import os
        #     if not os.path.exists('./models'):
        #         os.makedirs('./models')f
        #     #save just state dict for 0
        #     torch.save(agents[0].state_dict(), f'./models/agentwalls_{episode}.ckpt')
        #     exit(1)



        #if reward greater than 600 for hider_1 both for last and smoothed for last 5
        #every 100 epochs save the 2 models

        if episode % 100 == 0:
            torch.save(training_agents[0].state_dict(), f'./models/HIDER_SOLOTRAIN{episode}.ckpt')
        