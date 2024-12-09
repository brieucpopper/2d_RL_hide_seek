####Final project for DRL 8803 class

![random](https://github.com/user-attachments/assets/cb374134-8b6e-4ddc-b5e7-3dc3f8f8cd28)


This GIF is an example of our agents playing the game

The easiest way to get our code running is to have a PyTorch (ideally on GPU but not necessary) python 3.10 conda env, then pip install
 - PettingZoo (pip install pettingzoo)
 - wandb (pip install wandb)
 - pygame (pip install pygame)
 - imageio (pip install imageio)


The folders here are
 - old (old code for our initial experiments, not useful in the end)
 - PARALLEL : all the useful code is there because in the end we chose to go with a parallel pettingzoo environment, see the README.md there for details


Quickstart : once your python env is activated run (from ./2d_RL_hide_seek)

```python PARALLEL/train_specific_PPO.py```

This runs the experiments training each agent with an independent policy. To choose which agent to train, you can set the corresponding variable to True in IS_TRAINING at the beginning of this file.
To train the agents using a pretrained policy, set INITIALIZATION to the correct path for the corresponding agent.

To train the agents with a shared policy, you can run the notebook PPO_centralized.ipynb (from ./2d_RL_hide_seek/PARALLEL)

This runs the experiments training a policy for each team. To choose which team to train, you can set the corresponding variable to True in IS_POLICY_TRAINING at the beginning of this file.
To train the agents using a pretrained policy, set INITIALIZATION to the correct path for the corresponding agents.
Set SAVE_PRED_POL and SAVE_HIDER_POL to True/False depending on whether you want to save checkpoints.

If you do not want to log the graphs to weights and biases live (wandb) you can just ignore it.

=======
If you don't want to log the graphs to weights and biases live (wandb) you can just ignore it


Finally there is code in branch ```offline``` and documentation that goes with there
