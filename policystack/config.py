import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass, field, MISSING

from math.advantage import gae
from math.objective import clipped_surrogate_with_entropy, critic_mse

from __future__ import annotations
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from training import TrainingContext
    

class DynamicTerm:
    """
    Modifies configuration variables according to training context
    """
    def __init__(self, callback: Callable) -> None:
        self.callback = callback
        
    def get(self, ctx: TrainingContext) -> Any:
        # several potential arguments 
        return self.callback(ctx)
    
    # the following are factory methods with common numerical annealing functions
    
    @classmethod
    def linear_decay(cls, start: float, end: float) -> DynamicTerm: # type: ignore
        linear_fn = lambda ctx: start + (end - start) * ctx.progress
        return cls(linear_fn)
    
    @classmethod
    def expo_decay(cls, start: float, end: float) -> DynamicTerm: # type: ignore
        expo_fn = lambda ctx: start * (start / end) ** ctx.progress
        return cls(expo_fn)
    
    @classmethod
    def step_decay(cls, start: float, end: float, step: float, latency: int) -> DynamicTerm: # type: ignore
        step_fn = lambda ctx: max(start - (ctx.steps // latency) * step, end)
        return cls(step_fn)
    
    @classmethod
    def constant(cls, value: Any) -> DynamicTerm: # type: ignore
        const_fn = lambda ctx: value
        return cls(const_fn)
    


@dataclass
class Config:
    # communication bridge; contains training context
    ctx: TrainingContext
    
    def __getattribute__(self, name: str):
        attr = getattr(self, name)
        # get updated attribute value
        return attr.get(self.ctx)



@dataclass
class PPOConfig(Config):
    """
    Config template for PPO
    """
    # network architecture parameters
    # for the state-action function, the return value of forward() must
    # be keyed {"continuous": ..., "discrete": ...}
    # else, all actions will be assumed to be continous and may impose downstream errors
    actor: nn.Module
    critic: nn.Module
    # assumes that actor/critic are trained seperately
    # i.e. no shared backbone
    actor_op: optim.Optimizer # note that learning rate scheduling is done within the optimizers; other curriculum
    critic_op: optim.Optimizer
    
    # enviornment must follow gymnassium convention
    # step(action) -> (obs, reward, term, trunc, info)
    # reset(seed=None) -> (obs, info)
    environment = field(default=MISSING)
    
    # number of times transitions from each rollout are iterated over
    epochs: int = 10
    # number of collect -> train cycles
    iterations: int = 200
    batch_size: int = 64
    # number of steps collected in rollout phase
    rollout_length: int = 1024
    
    # alloted ratio-difference between the target policy and trained policy
    # prevents caatstrophic poicy collapse by limited the amount the policy can learn in on cycle
    #clipping_param: float = 0.2
    
    policy_objective_fn: Callable = clipped_surrogate_with_entropy
    advantage_fn: Callable = gae
    critic_loss_fn: Callable = critic_mse
    
    policy_objective_params: dict[str, Any] = field(default={
        "clipping_param": 0.2, "entropy_coef": 0.01
    })
    advantage_params: dict[str, Any] = field(default={
        "discount_factor": 0.99, "gae_decay": 0.98
    })
    critic_loss_fn: dict[str, Any] = field(default={})
    
    # enables the use of a single optimizer on a weighted sum of the policy and value objectives; use with a shared backbone
    # otherwise, two 
    #couple_objectives: bool = True
    # compresses the range of designated variance outputs to (0, inf)
    #exponentiate_variance: bool = True
    
    

@dataclass
class DQNConfig:
    net: nn.Module = field(default=MISSING)
    # enables exploration in the dqn; policy with a probability of epsilon selects a random action
    epsilon_fn: Callable | float = field(
        default=lambda t: max(0.01, 1.0 - t / 100_000)
    )
    
    # number of collection + refinement cycles
    iterations: int = field(default=100_000)
    # define the ratio between environment and gradient update steps
    n_collections_per_iter: int = field(default=1)
    n_refinements_per_iter: int = field(default=1)
    
    batch_size: int = field(default=64)