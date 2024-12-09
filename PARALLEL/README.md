## parallel.py defines the PettingZoo parallel env

the grid is one hot encoded :
dim 0 is walls
dim 1 is pred_1
dim 2 is pred_2
then dim 3 and 4 : hider_1 and hider_2

you can change the grid_size
you can observe the play with 


```
env = parallel_env(render_mode='human',grid_size=7)

 observations, infos = env.reset()
 while env.agents:
 this is where you would insert your policy
     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
     observations, rewards, terminations, truncations, infos = env.step(actions)
 env.close()
```

Its best to watch any play with watch_play_and_stats.py
Make sure the reward is setup as you want !


## train_specific_PPO.py

Used to train a single agent, or multiple agents with PPO.
You can specify if each agent should be
 - training (from random init)
 - random (random actions)
 - follow a pre-defined checkpoint (with frozen weights)

The only variables that need to be modified are delimited in the code clearly.
## watch play and stats.py

Load policies, or random agents.

Generates a GIF by default
don't forget to set

GRIDSIZE
the environment
the policies you want to see play in the GIF

## movable_wall_parallel.py


same as parallel but with an added movable wall !

```
def input_to_action(input):
    if input == 'd':
        return 1 #right
    elif input == 'a':
        return 2 #left
    elif input == 's': 
        return 3 # down
    elif input == 'w':
        return 4 # up
    else:
        return 0

env = parallel_env(render_mode='human',grid_size=8,walls=True)

observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions['pred_1'] = input_to_action(input("enter pred_1 move (w a s d) then enter : \n"))
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
env.close()
```

You can play as pred_1 with the above code which helps test out the wall !

## centralized scripts

These scripts are useful for the experiments where there is only one policy for the two teams.
This includes 
 - watch_play_and_stats_centralized.py (and notebook though easier to work with .py)
 - PPO_centralized.ipynb

