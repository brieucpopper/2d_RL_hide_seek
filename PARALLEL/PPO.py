import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Flatten(),
        )
        self.actor = self._layer_init(nn.Linear(256, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(256, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x[:,[1,3]] / 1.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x[:,[1,3]]/1.0)
        
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    #obs = obs.transpose(0, -1, 1, 2)
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
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32

    max_cycles = 250
    total_episodes = 450

    """ ENV SETUP """
    import parallel
    env = parallel.parallel_env(render_mode=None,grid_size=7)

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.0005, betas=(0.5, 0.998), eps=1e-5)


    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    
    rb_obs = torch.zeros((max_cycles, num_agents, 5,7,7)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    ep_returns = []
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

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                
                #make actions random except the first one
                #actions is a tensor of size (num_agents,)
                actions[1:] = torch.randint(0, num_actions, (num_agents-1,))
                #print(actions)
                
                randommm = False
                if np.random.rand() < 0.00001:
                    randommm = True
                    print(f'obs: {obs}')
                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )
                if randommm:
                    print(f'nex_obs: {next_obs}')
                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(5):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
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

                # normalize advantaegs
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

                #if episode more than 200, ent_coef can take value 0.1
                if episode > 200:
                    ent_coef = 0.2
                
                if episode > 300:
                    ent_coef = 0.1

                optimizer.zero_grad()
                loss.backward()
                #clip gradient 
                #torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        
        ep_returns.append(total_episodic_return)
        #print(ep_returns)

        #if last ep return is more than -18 AND mean on last 10 is more than -18, save
        if np.mean(ep_returns[-1]) > 4444 and np.mean(ep_returns[-10:]) > 4444:
            torch.save(agent.state_dict(), "agent.pth")
            print("Saved agent")
            raise
        if episode % 5 == 0:
            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(total_episodic_return)}")
            
            #print mean of last 50 episodes
            if episode > 50:
                print(f"Mean of last 50 episodes: {np.mean(ep_returns[-50:])}")

            
            print(f"Episode Length: {end_step}")
            print("")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            print("\n-------------------------------------------\n")

    """ RENDER THE POLICY """

    import parallel
    env = parallel.parallel_env(render_mode="human",grid_size=7)

    agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                actions[1:] = torch.randint(0, num_actions, (num_agents-1,))
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]