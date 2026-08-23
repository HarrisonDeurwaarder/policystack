import torch
import torch.nn as nn

import copy

from policystack.utils.buffers import Replay
from policystack.config import DynamicTerm
from policystack.training import ValueBasedTrainer

from dataclasses import dataclass, field
from typing import Callable


class DQN(nn.Module):
    
    def __init__(self, config: DQNConfig) -> None:
        super().__init__()
        self.config = config
        self.net = config.net
        
    
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return super().__call__(obs)
        
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        out = self.policy(obs) # (B, E, A)
        # make the action distributions
        self.config.action_manager.make_dists(out)
        # return a sampled action
        return self.sample_action(deterministic)
    
    
    def q_value(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        idx = self.forward(obs, deterministic)
        # attain q-value by indexing dist parameters
        qval = self.config.action_manager.action_params[..., idx]
        return qval
        
        
    def sample_action(self, deterministic: bool = False) -> torch.Tensor:
        # sample using the current distribution
        return self.config.action_manager.sample(deterministic) # (B, E)
    
    
    def entropy(self) -> torch.Tensor:
        return self.config.action_manager.entropy() # (B, E)
    
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.config.action_manager.log_prob(action) # (B, E, A)
    
    
    def get_values(self, obs: torch.Tensor) -> torch.Tensor:
        value = self.value(obs)
        return value # (B, E, A)
    
    
    
class DQNTrainer(ValueBasedTrainer):
    def _pre_training(self) -> None:
        # instantiate replay buffer
        self.replay = Replay(["next_obs", "q_values", "rewards", "dones"])
        obs, _ = self.env.reset()
        
        # collect preliminary samples
        # no policy refinement
        while len(self.replay) < self.config.warmup:
            # register a distribution + sample from that distribution
            action = self.algorithm(obs)
            # index q value
            q_val = self.algorithm.action_manager.action_params[..., action]
            next_obs, reward, term, trunc, _ = self.env.step(action)
            self.replay.stage(
                {"next_obs": next_obs, "q_values": q_val, "rewards": reward, "dones": term | trunc}
            )
            # next_obs => obs for next step
            obs = next_obs
        # establish target policy
        self._update_frozen_policy()
        
        
    def _collect_transitions(self) -> None:
        action = self.dqn(self.replay.staged["obs"])
        # save "prior" obs
        next_obs, reward, term, trunc, _ = self.env.step(action)
        self.replay.stage(
            {"next_obs": next_obs, "actions": action, "rewards": reward, "dones": term | trunc}
        )
        self.replay.commit()
        # restage next obs for subsequent step
        self.replay.stage({"obs": next_obs})
        
        
    def _gradient_update(self, batch: dict[str, torch.Tensor]) -> None:
        # update policy
        self.config.op.zero_grad()
        # compute current distributions
        loss = self.config.loss_fn(
            reward=batch["rewards"],
            value=batch["q_values"],
            next_value=self.target_policy(batch["next_obs"]),
            done=batch["dones"],
            **self.config.loss_params,
        )
        loss.backward()
        self.config.op.step()
        
        
    def _update_frozen_policy(self) -> None:
        # update at frequency
        if self.state.learning_steps % self.config.target_update_interval:
            self.target_policy = DQN(self.algorithm.config) # algorithm is a shell for the net argument's parameters
            # set for eval only
            self.target_policy.requires_grad(False)
            self.target_policy.eval()
                
                
                
@dataclass
class DQNConfig:
    net: nn.Module
    # enables exploration in the dqn; policy with a probability of epsilon selects a random action
    epsilon_fn: DynamicTerm | float = 0.05
    
    # number of collection + refinement cycles
    iterations: int = 100_000
    # define the ratio between environment and gradient update steps
    collection_freq: DynamicTerm | int = 1
    refinement_freq: DynamicTerm | int = 1
    
    batch_size: DynamicTerm | int = 64
    
    target_update_interval: DynamicTerm | int = 8_000 # gradient updates