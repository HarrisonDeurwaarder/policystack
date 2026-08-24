import torch
import torch.nn as nn
import torch.optim as optim

import copy

from policystack.utils.buffers import Replay
from policystack.utils.config import DynamicTerm
from policystack.training import ValueBasedTrainer
from policystack.managers.actions import ActionManager
from policystack.math.objective import msbe

from dataclasses import dataclass, field
from typing import Callable, Any


class DQN(nn.Module):
    
    def __init__(self, config: DQNConfig) -> None:
        super().__init__()
        self.config = config
        self.net = config.net
        
    
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return super().__call__(obs)
        
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        out = self.net(obs) # (B, E, A)
        # make the action distributions
        self.config.action_manager.make_dists(out)
        # return a sampled action
        return self.sample_action(deterministic)
    
    
    def q_value(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        idx = self.forward(obs, deterministic)
        # attain q-value by indexing dist parameters
        return self.sample_action(deterministic=True)[..., idx]
        
        
    def sample_action(self, deterministic: bool = False) -> torch.Tensor:
        # sample using the current distribution
        return self.config.action_manager.sample(deterministic) # (B, E)
    
    
    def entropy(self) -> torch.Tensor:
        return self.config.action_manager.entropy() # (B, E)
    
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.config.action_manager.log_prob(action) # (B, E, A)
    
    
    
class DQNTrainer(ValueBasedTrainer):
    def _pre_training(self) -> None:
        # instantiate replay buffer
        self.replay = Replay(["obs", "next_obs", "q_values", "rewards", "dones"])
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
            self.replay.commit()
            # next_obs => obs for next step
            obs = next_obs
        # establish target policy
        self._update_frozen_policy()
        
        
    def _collect_transitions(self) -> None:
        action = self.algorithm(self.replay.staged["obs"])
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
        self.target_policy(batch["next_obs"])
        loss = self.config.loss_fn(
            reward=batch["rewards"],
            value=batch["q_values"],
            next_value=self.target_policy.q_value(),
            done=batch["dones"],
            **self.config.loss_params,
        )
        loss.backward()
        self.config.op.step()
        
        
    def _update_frozen_policy(self) -> None:
        # update at frequency
        if self.state.learning_steps % self.config.target_update_interval == 0:
            self.target_policy = copy.deepcopy(self.algorithm) # algorithm is a shell for the net argument's parameters
            # set for eval only
            self.target_policy.eval()
                
                
                
@dataclass
class DQNConfig:
    net: nn.Module
    # enables exploration in the dqn; policy with a probability of epsilon selects a random action
    epsilon_fn: DynamicTerm | float = 0.05
    # all raw logits pass through the action manager, then are divided into distributions specified by ActionTerms
    # and are recombined into action values, entropy, or lob probs
    action_manager: ActionManager
    
    # gradient update fields
    op: optim.Optimizer
    loss_fn: Callable = msbe
    loss_params: dict[str, Any] = field(default_factory=dict)
    
    # number of warmup steps
    warmup: int = 8_000
    # number of collection + refinement cycles
    iterations: int = 100_000
    # define the ratio between environment and gradient update steps
    collection_freq: DynamicTerm | int = 1
    refinement_freq: DynamicTerm | int = 1
    
    batch_size: DynamicTerm | int = 64
    
    target_update_interval: DynamicTerm | int = 8_000 # gradient updates