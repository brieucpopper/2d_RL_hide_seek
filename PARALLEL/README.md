## parallel.py defines the PettingZoo parallel env

the grid is one hot encoded :
dim 0 is walls
dim 1 is pred_1
dim 2 is pred_2
then dim 3 and 4 : hider_1 and hider_2

you can change the grid_size
you can observe the play with 


***
env = parallel_env(render_mode='human',grid_size=7)

 observations, infos = env.reset()

 while env.agents:
 this is where you would insert your policy
     actions = {agent: env.action_space(agent).sample() for agent in env.agents}

     observations, rewards, terminations, truncations, infos = env.step(actions)
 env.close()
***

Make sure the reward is setup as you want !


##  weights

Stores the checkpoints for policies
A checkpoint will only work if grid_Sizes match of course.
It's basically the weights for the neural nets in the PPO algorithm

## train_specific_PPO.py

Used to train a single agent, or multiple agents with PPO.
You can specify if each agent should be
 - training (from random init)
 - random (random actions)
 - follow a pre-defined checkpoint (with frozen weights)


## watch play and stats.py

Load policies, or random agents.

you can watch the policies rendered with pygame, and get stats printed to terminal (like average rewards)