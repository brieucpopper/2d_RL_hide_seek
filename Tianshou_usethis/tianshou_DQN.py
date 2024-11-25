import os
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

import tianshou_env

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Union


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        print(' setup with state_shape:', state_shape, ' action_shape:', action_shape)
        super().__init__()
        
        self.model = nn.Sequential(
            # First conv layer
            nn.Conv2d(state_shape[0], 64, 7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Second conv layer
            nn.Conv2d(64, 64, 7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.MaxPool2d(2, 2),
            
            # Flatten and FC layer
            nn.Flatten(),
            nn.Linear(64 * state_shape[1]//2 * state_shape[2]//2, action_shape)
        )
        
    

    def forward(self, obs, state=None, info={}):
        

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=DEVICE)
        batch = obs.shape[0]
        logits = self.model(obs)
        return logits, state

def _get_agents(
    agent_learn1: Optional[BasePolicy] = None,
    agent_learn2: Optional[BasePolicy] = None,
    agent_opponent1: Optional[BasePolicy] = None,
    agent_opponent2: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    if agent_learn1 is None:
        # model
        net = CNN_Net(
            (observation_space.shape[0], observation_space.shape[1], observation_space.shape[2]),
            env.action_space.n,
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    

        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn1 = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=1,
            target_update_freq=1000,
            is_double = True,

        )
    if agent_learn2 is None:
        agent_learn2 = RandomPolicy()
    

    if agent_opponent1 is None:
        agent_opponent1 = RandomPolicy()

    if agent_opponent2 is None:
        agent_opponent2 = RandomPolicy()

    agents = [agent_learn1,agent_learn2,agent_opponent1,agent_opponent2]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(tianshou_env.env())


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    TRAINING_NUM = 1
    train_envs = DummyVectorEnv([_get_env for _ in range(TRAINING_NUM)])
    test_envs = DummyVectorEnv([_get_env for _ in range(TRAINING_NUM)])

    # seed
    seed = 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    for s in range(seed,seed+TRAINING_NUM):
        train_envs.seed(s)
        test_envs.seed(s)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(100_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * TRAINING_NUM)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ttt", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "ttt", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 500

    def train_fn(epoch, env_step):
        eps = 1
        if epoch < 5:
            eps = 1
        else:
            eps = 0.999 ** (env_step - 4)
        policy.policies[agents[0]].set_eps(eps)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0)

    def reward_metric(rews):
 
        return rews[:, 0]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=4000,
        step_per_collect=1,
        episode_per_test=2,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[0]])")
