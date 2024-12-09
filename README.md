Final project for DRL 8803 class

![random](https://github.com/user-attachments/assets/cb374134-8b6e-4ddc-b5e7-3dc3f8f8cd28)


This GIF is an example of our agents playing the game

The easiest way to get our code running is to have a PyTorch (ideally on GPU but not necessary) python 3.10 conda env, then pip install
 - PettingZoo (pip install pettingzoo)
 - wandb (pip install wandb)
 - pygame (pip install pygame)


The folders here are
 - old (old code for our initial experiments, not useful in the end)
 - PARALLEL : all the useful code is there because in the end we chose to go with a parallel pettingzoo environment, see the README.md there for details


Quickstart : once your python env is activated run (from ./2d_RL_hide_seek)

```python PARALLEL/train_specific_PPO.py```