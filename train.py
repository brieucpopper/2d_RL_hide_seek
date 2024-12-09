

import gym
#import pybullet_envs
import numpy as np
from collections import deque
import torch
import time
import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
from agent import CQLSAC
from human_policy import create_buffer
from env_clean import PettingZooGridWorld

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-SAC-discrete", help="Run name, default: CQL-SAC")
    #parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    #parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=25, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--env_size", type=int, default=16, help="Size of the environment, default: 16")
    parser.add_argument("--n_walls", type=int, default=35, help="Number of walls in the environment, default: 4")
    parser.add_argument("--distinct_policy", type=bool, default=False, help="Use different policy for pred and prey, default: False")
    parser.add_argument("--randomness", type=float, default=0.1, help="Randomness of the prey, default: 0.1")
    parser.add_argument("--mixed", type=bool, default=False, help="Introduce randomness into the buffer, default: False")
    args = parser.parse_args()
    return args

#solve parser issues with bool
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    #env = gym.make(config.env)
    
    #env.seed(config.seed)
    #env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "mps"
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    config.mixed = str2bool(config.mixed)
    config.distinct_policy = str2bool(config.distinct_policy)
    with wandb.init(project="CQL", name=config.run_name, config=config):
        
        #agent = CQLSAC(state_size=env.observation_space.shape[0],
        #                 action_size=env.action_space.n,
        #                 device=device)
        if not config.distinct_policy:
            agent = CQLSAC(state_size=3,action_size=1,device='mps',env_size=config.env_size)
            wandb.watch(agent, log="gradients", log_freq=10)
        else:
            agent_prey = CQLSAC(state_size=3,action_size=1,device='mps',env_size=config.env_size)
            agent_pred = CQLSAC(state_size=3,action_size=1,device='mps',env_size=config.env_size)
            wandb.watch(agent_pred, log="gradients", log_freq=10)
            wandb.watch(agent_prey, log="gradients", log_freq=10)

        #wandb.watch(agent, log="gradients", log_freq=10)

        #buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        
        #collect_random(env=env, dataset=buffer, num_samples=1000)
        if not config.distinct_policy:
            buffer = create_buffer(config.env_size,config.n_walls)
            loss_logs = []
        else:
            buffer_pred = create_buffer(config.env_size,config.n_walls,randomness_pred=0.01,randomness_prey=0.5,agentType="pred",mixed=config.mixed) 
            buffer_prey = create_buffer(config.env_size,config.n_walls,randomness_pred = 0.5,randomness_prey=0.01,agentType="prey",mixed=config.mixed)
            loss_logs_pred = []
            loss_logs_prey = []
        #if config.log_video:
        #    env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)
        rewards_log = []
        
        for i in range(1, config.episodes+1):
            #state = env.reset()
            episode_steps = 0
            #rewards = 0
            #while True:
            for k in range(10):
                #action = agent.get_action(state)
                #steps += 1
                #next_state, reward, done, _ = env.step(action)
                #buffer.add(state, action, reward, next_state, done)
                if not config.distinct_policy:
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
                    loss_logs.append(policy_loss)
                #state = next_state
                else:
                    policy_loss_pred, alpha_loss_pred, bellmann_error1_pred, bellmann_error2_pred, cql1_loss_pred, cql2_loss_pred, current_alpha_pred, lagrange_alpha_loss_pred, lagrange_alpha_pred = agent_pred.learn(steps, buffer_pred.sample(), gamma=0.99)
                    policy_loss_prey, alpha_loss_prey, bellmann_error1_prey, bellmann_error2_prey, cql1_loss_prey, cql2_loss_prey, current_alpha_prey, lagrange_alpha_loss_prey, lagrange_alpha_prey = agent_prey.learn(steps, buffer_prey.sample(), gamma=0.99)
                    loss_logs_pred.append(policy_loss_pred)
                    loss_logs_prey.append(policy_loss_prey)
                    ######## Online reward
                    
                    
                    
                    
                #rewards += reward
                episode_steps += 1
                #if done:
                #    break
            
            
            env = PettingZooGridWorld(config.env_size,walls = True, num_walls = config.n_walls)
            env.reset()
            cum_reward = None
            #time evaluation
            t = time.time()
            for _ in range(200):
                actions = {}
                for Agent in env.agents:
                    state = env.get_observations()[Agent]
                    if not config.distinct_policy:
                        action = agent.actor_local.get_det_action((torch.tensor(state,dtype=torch.float32).unsqueeze(0).to('mps'))).item()
                    else:
                        if 'prey' in Agent:
                            action = agent_prey.actor_local.get_det_action((torch.tensor(state,dtype=torch.float32).unsqueeze(0).to('mps'))).item()
                        else:
                            action = agent_pred.actor_local.get_det_action((torch.tensor(state,dtype=torch.float32).unsqueeze(0).to('mps'))).item()
                    actions[Agent] = action
                observations, rewards, dones, infos = env.step(actions)
                if cum_reward is None:
                    cum_reward = rewards
                else:
                    for Agent in env.agents:
                        cum_reward[Agent] += rewards[Agent] 
            print("Time for 200 steps: ",time.time()-t)
            rewards_log.append(cum_reward)
            
            rewards = 0
            average10.append(rewards)
            total_steps += episode_steps
            if not config.distinct_policy:
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
            else:
                #rewards_log[-1]['prey_1'],rewards_log[-1]['pred_1'], | Reward Prey: {} | Reward Pred: {} 
                print("Episode: {} | Reward Prey: {} | Reward Pred: {} | Polciy Loss Pred: {} | Polciy Loss Prey: {} | Steps: {}".format(i, rewards_log[-1]['prey_1'],rewards_log[-1]['pred_1'],policy_loss_pred,policy_loss_prey, steps,))
            
            if i % config.save_every == 0:
                if not config.distinct_policy:
                    save(config, save_name="CQL-SAC-discrete", model=agent.actor_local, wandb=wandb, ep=0)
                    #save rewards_log
                    np.save("rewards{}.npy".format(config.run_name),rewards_log)
                    np.save("losses{}.npy".format(config.run_name),loss_logs)
                else:
                    save(config, save_name="CQL-SAC-discretePred", model=agent_pred.actor_local, wandb=wandb, ep=0)
                    save(config, save_name="CQL-SAC-discretePrey", model=agent_prey.actor_local, wandb=wandb, ep=0)
                    #save rewards_log
                    np.save("rewards{}.npy".format(config.run_name),rewards_log)
                    np.save("losses_pred{}.npy".format(config.run_name),loss_logs_pred)
                    np.save("losses_prey{}.npy".format(config.run_name),loss_logs_prey)
                    
if __name__ == "__main__":
    config = get_config()
    train(config)
