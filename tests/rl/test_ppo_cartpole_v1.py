from policystack.rl.ppo import *
from policystack.managers.actions import *

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym



class IntermediateEnv:
    def __init__(self):
        self.env: gym.Env = gym.make("CartPole-v1")
        
    def reset(self):
        obs, info = self.env.reset(seed=42)
        return torch.from_numpy(obs).float(), info
    
    def step(self, action: torch.Tensor):
        obs, reward, term, trunc, info = self.env.step(
            action.int().detach().cpu().item()
        )
        # reset if needed
        if term or trunc:
            obs, info = self.env.reset(seed=42)
        return torch.from_numpy(obs), torch.tensor([reward]), torch.tensor([term]), torch.tensor([trunc]), info


def test_ppo():
    # deep model architectures
    actor = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    critic = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # configs
    act_config = ActionConfig(
        terms=[
            BernoulliAction(n_logits=1)
        ]
    )
    ppo_config = PPOConfig(
        actor=actor,
        critic=critic,
        action_config=act_config
    )
    trainer_config = PPOTrainerConfig(
        actor_op=optim.Adam(params=actor.parameters()),
        critic_op=optim.Adam(params=actor.parameters()),
        environment=IntermediateEnv(),
    )
    
    ppo = PPO(config=ppo_config)
    trainer = PPOTrainer(config=trainer_config, algorithm=ppo)
    
    trainer.train()