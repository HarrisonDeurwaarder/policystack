import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass, field, MISSING
from typing import Callable, Any

from math.advantage import gae
from math.objective import clipped_surrogate_with_entropy, critic_mse


@dataclass
class PPOConfig:
    """
    Config template for PPO
    """
    # network architecture parameters
    # for the state-action function, the return value of forward() must
    # be keyed {"continuous": ..., "discrete": ...}
    # else, all actions will be assumed to be continous and may impose downstream errors
    actor: nn.Module = field(default=MISSING)
    critic: nn.Module = field(default=MISSING)
    # assumes that actor/critic are trained seperately
    # i.e. no shared backbone
    actor_op: optim.Optimizer = field(default=MISSING)
    critic_op: optim.Optimizer = field(default=MISSING)
    
    # enviornment must follow gymnassium convention
    # step(action) -> (obs, reward, term, trunc, info)
    # reset(seed=None) -> (obs, info)
    environment = field(default=MISSING)
    
    epochs: int = field(default=10)
    iterations: int = field(default=200)
    batch_size: int = field(default=64)
    rollout_length: int = field(default=1024)
    
    #entropy_coefficient: float = field(default=1e-3)
    discount_factor: float = field(default=0.99)
    gae_decay: float = field(default=0.98)
    clipping_param: float = field(default=0.2)
    
    policy_objective_fn: Callable = field(default=clipped_surrogate_with_entropy)
    advantage_fn: Callable = field(default=gae)
    critic_loss_fn: Callable = field(default=critic_mse)
    
    policy_objective_params: dict[str, Any] = field(default={
        "clipping_param": 0.2, "entropy_coef": 0.01
    })
    advantage_params: dict[str, Any] = field(default={
        "discount_factor": 0.99, "gae_decay": 0.98
    })
    critic_loss_fn: dict[str, Any] = field(default={})
    
    # enables the use of a single optimizer on a weighted sum of the policy and value objectives; use with a shared backbone
    # otherwise, two 
    couple_objectives: bool = True
    # compresses the range of designated variance outputs to (0, inf)
    exponentiate_variance: bool = True