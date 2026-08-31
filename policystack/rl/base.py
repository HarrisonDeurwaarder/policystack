from __future__ import annotations
from typing import TYPE_CHECKING

from policystack.managers.actions import ActionManager

import torch
import torch.nn as nn

from abc import ABC, abstractmethod



class RLAlgorithm(nn.Module, ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.action_manager = ActionManager(config.action_config)
        
        
    def __call__(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return super().__call__(obs, deterministic)
    
    
    @property
    def logits(self) -> torch.Tensor:
        return self.action_manager.logits()
    
    
    @property
    def entropy(self) -> torch.Tensor:
        return self.action_manager.entropy()
    
    
    def sample_action(self, deterministic: bool = False) -> torch.Tensor:
        return self.action_manager.sample(deterministic=deterministic)
    
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_manager.log_prob(actions=action)
    
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        ...