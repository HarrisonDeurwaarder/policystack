import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass, field, MISSING

from math.advantage import gae
from math.objective import clipped_surrogate_with_entropy, critic_mse

from __future__ import annotations
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from training import TrainingState
    
    
def resolve(v: DynamicTerm) -> Any:
    # access as a regular variable, unless it is a DynamicTerm then access with updated value
    # this should be used to access any variables that may have reason to follow a curriculum
    return v.get() if isinstance(v, DynamicTerm) else v
    

class DynamicTerm:
    """
    Modifies configuration variables according to training context
    It is good practice to instantiate a DynamicTerm under a TrainingState object in order to avoid specifying the training state, and for clarity
    """
    def __init__(self, callback: Callable, state: TrainingState) -> None:
        self.callback = callback
        self.state = state
        
    def get(self) -> Any:
        # several potential arguments 
        return self.callback(self.state)
    
    # the following return common callback functions for numerical terms
    
    @staticmethod
    def linear_decay(start: float, end: float) -> Callable:
        linear_fn = lambda ctx: start + (end - start) * ctx.progress
        return linear_fn
    
    @staticmethod
    def expo_decay(start: float, end: float) -> Callable:
        expo_fn = lambda ctx: start * (start / end) ** ctx.progress
        return expo_fn
    
    @staticmethod
    def step_decay(start: float, end: float, step: float, latency: int) -> Callable:
        step_fn = lambda ctx: max(start - (ctx.steps // latency) * step, end)
        return step_fn
    
    @staticmethod
    def constant(value: Any) -> Callable:
        const_fn = lambda ctx: value
        return const_fn