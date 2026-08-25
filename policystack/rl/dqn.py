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
        
    
    def __call__(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return super().__call__(obs, deterministic)
        
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Assembles and returns an action index"""
        out = self.net(obs) # (B, E, A)
        # make the action distributions
        self.config.action_manager.make_dists(out)
        # return a sampled action
        return self.sample_action(deterministic)
    
    
    def q_values(self) -> torch.Tensor:
        """Assumes the correct distribution to be assembled; returns q-values as given by the network"""
        # sample using the current distribution
        return self.config.action_manager.logits() # (B, E, L)
        
        
    def sample_action(self, deterministic: bool = False) -> torch.Tensor:
        """Assumes the distribution to be assembled; returns the action onehot"""
        return self.config.action_manager.sample_action(deterministic) # (B, E, A)
    
    
    def entropy(self) -> torch.Tensor:
        return self.config.action_manager.entropy() # (B, E)
    
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.config.action_manager.log_prob(action) # (B, E, A)
    
    
    
class DQNTrainer(ValueBasedTrainer):
    def _pre_training(self) -> None:
        # instantiate replay buffer
        self.replay = Replay(["obs", "next_obs", "q_values", "rewards", "dones"], length=self.config.buffer_size)
        obs, _ = self.env.reset()
        self.replay.stage({"obs": obs})
        
        # collect preliminary samples
        # no policy refinement
        while len(self.replay) < self.config.warmup:
            self._collect_transitions()
        # establish target policy
        self._update_frozen_policy()
        
        
    def _collect_transitions(self) -> None:
        idx = self.algorithm(self.replay.staged["obs"])
        # index q-value
        # usable q-values must not have exploration applied
        action_qval = self.algorithm.q_values(deterministic=True)[..., idx]
        next_obs, reward, term, trunc, _ = self.env.step(idx)
        # save "prior" obs
        self.replay.stage(
            {"next_obs": next_obs, "q_values": action_qval, "rewards": reward, "dones": term | trunc}
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
            next_value=self.target_policy.q_values(deterministic),
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
    # all raw logits pass through the action manager, then are divided into distributions specified by ActionTerms
    # and are recombined into action values, entropy, or lob probs
    action_manager: ActionManager
    
    # gradient update fields
    op: optim.Optimizer
    loss_fn: Callable = msbe
    loss_params: dict[str, Any] = field(default_factory=dict)
    
    # enables exploration in the dqn; policy with a probability of epsilon selects a random action
    epsilon_fn: DynamicTerm | float = 0.05
    
    # number of warmup steps
    warmup: int = 8_000
    # number of collection + refinement cycles
    iterations: int = 100_000
    # define the ratio between environment and gradient update steps
    collection_freq: DynamicTerm | int = 1
    refinement_freq: DynamicTerm | int = 1
    
    buffer_size: DynamicTerm | int = 1_000_000
    batch_size: DynamicTerm | int = 64
    
    target_update_interval: DynamicTerm | int = 8_000 # gradient updates