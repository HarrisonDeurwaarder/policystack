import torch
import torch.nn as nn
import torch.nn.functional as F

from config import resolve

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import DynamicTerm


def clipped_surrogate_objective(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clipping_param: float = 0.2 | DynamicTerm
) -> torch.Tensor:
    """
    Computes the policy objective at a state for PPO
    
    Args:
        log_prob (torch.Tensor): log probability of the chosen action being chosen in the current policy (B, E)
        old_log_prob (torch.Tensor): log probability of the chosen action being chosen in the old policy (B, E)
        advantage (torch.Tensor): benefit of the policy's choice over what was expected by the critic (B, E)
        clipping_param (float): amount of deviation from the old policy (expressed as a ratio of probabilities) that is accepted
        
    Returns:
        torch.Tensor: the objective (B, E)
    """
    # ratio between current policy and frozen policy aids in preventing excessive updates
    # logarithmic properties are used to smooth computations for extremely small probabilities
    ratio = torch.exp(log_prob - old_log_prob)
    # consider the benefits of the clipped surrogate objective in four cases
    # advantage is positive, ratio is clipped above 1 + epsilon: rate at which this good action is encouraged is limited to prevent a catastrophically large update
    # advantage is positive, ratio is clipped below 1 - epsilon: rate at which this good action is taken is limited currently and should be more common (i.e. policy can catch up)
    # advantage is negative, ratio is clipped above 1 + epsilon: rate at which this bad action is taken is excessive currently and should be less common (i.e. policy can catch up)
    # advantage is negative, ratio is clipped below 1 - epsilon: rate at which this bad action is discourage is limited to prevent a catastrophically large update
    objective = torch.min(
        advantage * ratio,
        advantage * torch.clip(
            ratio, 1 - resolve(clipping_param), 1 + resolve(clipping_param)
        )
    )
    return objective.mean() # reduce to scalar


def clipped_surrogate_with_entropy(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    entropy: float,
    clipping_param: float | DynamicTerm,
    entropy_coef: float | DynamicTerm,
) -> torch.Tensor:
    objective = clipped_surrogate_objective(
        log_prob=log_prob, old_log_prob=old_log_prob, advantage=advantage, clipping_param=clipping_param
    )
    # add the entropy term
    return objective + torch.mean(resolve(entropy_coef) * entropy)
    
    
def critic_mse(
    expected_value: torch.Tensor,
    old_expected_value: torch.Tensor,
    advantage: torch.Tensor
) -> torch.Tensor:
    """
    Computes the value loss at a state for AC algorithms

    Args:
        expected_value (torch.Tensor): critic expected value of the policy given the state (B, E)
        old_expected_value (torch.Tensor): old critic expected value of the policy given the state (B, E)
        advantage (torch.Tensor): benefit of the policy's choice over what was expected by the critic (B, E)

    Returns:
        torch.Tensor: the loss (B, E)
    """
    loss = F.mse_loss(
        expected_value,
        old_expected_value + advantage
    )
    return loss


def msbe(
    reward: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    done: torch.Tensor,
    discount_factor: float | DynamicTerm
) -> torch.Tensor:
    """
    Computes the mean squared bellman error (td loss) of a value prediction
    MSBE bootstraps a prediction using td error, and as such has low variance and high bias

    Args:
        reward (torch.Tensor): episode rewards of the policy (B, E)
        value (torch.Tensor): value function's expected value of the state or state-action (B, E)
        next_value (torch.Tensor): value function's expected value of the next state (B, E)
        done (torch.Tensor): boolean flags indicating whether an episode is done (1) or not (0) (B, E)
        discount_factor (float): discount factor

    Returns:
        torch.Tensor: the loss (B, E)
    """
    loss = F.mse_loss(value, reward + (1 - done) * resolve(discount_factor) * next_value)