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

class CustomNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        # Replace convolution layers with a fully connected architecture
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 256), nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, np.prod(action_shape))
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_ally1: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    agent_opponent2: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        net = CustomNet(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=2,
            target_update_freq=50,
        )

    if agent_opponent is None:
        agent_opponent = RandomPolicy()
    if agent_ally1 is None:
        agent_ally1 = RandomPolicy()
    if agent_opponent2 is None:
        agent_opponent2 = RandomPolicy()

    agents = [agent_learn, agent_ally1, agent_opponent, agent_opponent2]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(tianshou_env.env())


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ttt", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "ttt", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.999**env_step)
        if np.random.rand()<0.001:
            print(f"eps: {policy.policies[agents[0]].eps}")

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.0)

    def reward_metric(rews):
        return rews[:, 0]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
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
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")