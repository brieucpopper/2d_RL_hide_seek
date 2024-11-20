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


#import the env from ./env.py
from env import PettingZooGridWorld


MAX_STEPS = 1000
GRID_SIZE = 5

###############################################################################################
# RUN RANDOM POLICY : AT EACH TIME STEP EVERY AGENTS SAMPLES AN ACTION AT RANDOM

for it in range(10):  #we do this random policy 5 times to get a sense of the variance
    env = PettingZooGridWorld(GRID_SIZE)
    env.reset()
    

    cum_reward = None
    # Run the environment for 100 steps
    for _ in range(MAX_STEPS):
        # Get the current agent
        agent = env.agent_selector.next()
        # Take a random action for the current agent
        action = env.action_spaces[agent].sample()
        
        # Step the environment
        observations, rewards, dones, infos = env.step({agent: action})
        
        #cum_reward is a dict with agent names as keys
        #increase by rewards

        if cum_reward is None:
            cum_reward = rewards
        else:
            for agent in env.agents:
                cum_reward[agent] += rewards[agent]
        # Render the environment
        #env.render()
        
        # Wait for a short time to slow down the animation

    print(f"For iteration {it} of random policy, the cumulative reward for pred_1 is {cum_reward['pred_1']}")
###############################################################################################






want_to_do_training=False
from simple_dqn_for_env import DQNAgent, DQN
if want_to_do_training:
###############################################################################################
    # run DQN
    

    GRID_SIZE = 5
    def train_predators():
        env = PettingZooGridWorld(GRID_SIZE)

        # Create DQN agents for predators
        state_size = GRID_SIZE * GRID_SIZE + 1  # Grid size plus team indicator
        action_size = 5  # Number of possible actions
        NUM_POSSIBLE_THINGS = 5
        #wall,air,self, predator (seeker), prey (hider) coded as 0,1,2,3,4
        pred1_agent = DQNAgent(state_size, 5,GRID_SIZE,NUM_POSSIBLE_THINGS)
        pred2_agent = DQNAgent(state_size, 5,GRID_SIZE,NUM_POSSIBLE_THINGS)
        
        batch_size = 256
        episodes = 1000
        update_target_every = 5
        
        # Track metrics for plotting
        all_rewards = {
            'pred_1': [],
            'pred_2': []
        }
        running_loss = []
        
        for episode in range(episodes):
            observations = env.reset()
            episode_rewards = {agent: 0 for agent in env.agents}
            done = False
            step = 0
            
            while not done:
                actions = {}
                
                for agent in env.agents:
                    if 'pred' in agent:
                        # Use DQN for predators
                        dqn_agent = pred1_agent if agent == 'pred_1' else pred2_agent
                        state = observations[agent]
                        action = dqn_agent.act(state)
                        actions[agent] = action
                    else:
                        # Random actions for prey
                        actions[agent] = env.action_spaces[agent].sample()
                
                next_observations, rewards, dones, _ = env.step(actions)
                done = any(dones.values())
                
                # Store experiences for predators
                for agent in ['pred_1', 'pred_2']:
                    dqn_agent = pred1_agent if agent == 'pred_1' else pred2_agent
                    dqn_agent.remember(
                        observations[agent],
                        actions[agent],
                        rewards[agent],
                        next_observations[agent],
                        dones[agent]
                    )
                    episode_rewards[agent] += rewards[agent]
                
                observations = next_observations
                step += 1
                
                # Train predator agents
                pred1_agent.replay(batch_size)
                pred2_agent.replay(batch_size)
            
            # Update target networks periodically
            if episode % update_target_every == 0:
                pred1_agent.update_target_model()
                pred2_agent.update_target_model()

                #save model to ./weights
                #torch.save(pred1_agent.model.state_dict(), f"./weights/pred1_{episode}.pt")
                #torch.save(pred2_agent.model.state_dict(), f"./weights/pred2_{episode}.pt")
            
            # Store rewards for plotting
            all_rewards['pred_1'].append(episode_rewards['pred_1'])
            all_rewards['pred_2'].append(episode_rewards['pred_2'])
            
            if episode % 1 == 0:
                print(f"Episode: {episode}, Pred1 Reward: {episode_rewards['pred_1']:.2f}, "
                    f"Pred2 Reward: {episode_rewards['pred_2']:.2f}, Epsilon: {pred1_agent.epsilon:.2f}")
        
        return pred1_agent, pred2_agent, all_rewards


    pred1_agent, pred2_agent, all_rewards = train_predators()
    # Run the commented code above to train the predators. You can optionally save checkpoints as you go
    ###############################################################################################



GRID_SIZE=5

###############################################################################################
# Test a pre-saved policy
PATH_1 = './demo_weights/pred1_55.pt'
PATH_2 = './demo_weights/pred2_55.pt'
NUM_POSSIBLE_THINGS=5
pred1_agent = DQNAgent(GRID_SIZE * GRID_SIZE + 1, 5,GRID_SIZE,NUM_POSSIBLE_THINGS)
pred1_agent.model.load_state_dict(torch.load(PATH_1))

pred2_agent = DQNAgent(GRID_SIZE * GRID_SIZE + 1, 5,GRID_SIZE,NUM_POSSIBLE_THINGS)
pred2_agent.model.load_state_dict(torch.load(PATH_2))

env = PettingZooGridWorld(GRID_SIZE)
observations = env.reset()
done = False
env.render()

policy_rewards = {agent: 0 for agent in env.agents}
while not done:
    actions = {}

    
    for agent in env.agents:
        if 'pred' in agent:
            dqn_agent = pred1_agent if agent == 'pred_1' else pred2_agent
            state = observations[agent]
            action = dqn_agent.act(state, training=False)
            actions[agent] = action
            #make sure only the predators move according to policy
        else:
            actions[agent] = env.action_spaces[agent].sample()
            #move random
    
    observations, rewards, dones, _ = env.step(actions)
    for agent in env.agents:
        policy_rewards[agent] += rewards[agent]
    done = any(dones.values())
    env.render()
    #time.sleep(0.8)
print(f" with these loaded policies from {PATH_1} and {PATH_2}, the predators achieved a total reward of {policy_rewards['pred_1']:.2f} and {policy_rewards['pred_2']:.2f}")