import torch
import torch.nn as nn

from config import resolve

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import DynamicTerm


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    discount_factor: float | DynamicTerm,
    gae_decay: float | DynamicTerm,
) -> torch.Tensor:
    """
    Computes generalized advantage estimates

    Args:
        rewards (torch.Tensor): episode rewards of the policy (B, E)
        values (torch.Tensor): critic's expected values of the policy at each state (B, E)
        next_values (torch.Tensor): values at the next consecutive state (B, E)
        dones (torch.Tensor): boolean flags indicating whether an episode is done (1) or not (0) (B, E)
        discount_factor (float): discount factor
        gae_decay (float): gae decay

    Returns:
        torch.Tensor: advantage estimates (B, E)
    """
    # td residual component (low variance)
    # takes the difference between the netted rewards (bootstrapped with the next state to determine value) and the critic's expected value
    td_res = td_residual(rewards, values, next_values, dones, discount_factor)
    
    # recursive monte carlo component (low bias)
    # computes the discounted value of a finite episode directly
    adv_dims = rewards.shape()
    adv_dims[-1] += 1 # an additional dimension is added to ease code complexity
    advantages = torch.zeros(adv_dims)
    for t in range(rewards.size(-1) - 1, -1, -1): # iterate backwards from T to 0, inclusive
        advantages[..., t] = td_res[..., t] + resolve(discount_factor) * resolve(gae_decay) * advantages[..., t+1]
        
    return advantages[..., :-1]


def td_residual(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    discount_factor: float | DynamicTerm,
) -> torch.Tensor:
    """
    Computes the TD residual advantages
    
    TD residual is a low-variance method of advantage computation that compares the critic's expected value of
    the policy to a bootstrapped value in which the "ground truth" value is estimated using the reward and the
    critic's succeeding value estimation
    TD residual is, however, high-biased because of its reliance on the policy's own predictions to serve as the ground-truth

    Args:
        rewards (torch.Tensor): episode rewards of the policy (B, E)
        values (torch.Tensor): critic's expected values of the policy at each state (B, E)
        next_values (torch.Tensor): values at the next consecutive state (B, E)
        dones (torch.Tensor): boolean flags indicating whether an episode is done (1) or not (0) (B, E)
        discount_factor (float): discount factor

    Returns:
        torch.Tensor: advantage estimates (B, E)
    """
    advantages = rewards + resolve(discount_factor) * values * (1 - dones) - next_values[..., :-1]
    return advantages


def monte_carlo(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    discount_factor: float | DynamicTerm,
) -> torch.Tensor:
    """
    Computes the monte carlo advantages
    
    Monte carlo advantages are low-bias that compare the critic's expectation to the true value of a state given the policy
    Though, monte carlo advantages are high-variance and do not recieve any smoothing as GAE or TD residuals would,
    resulting in noisy advantage signals

    Args:
        rewards (torch.Tensor): episode rewards of the policy (B, E)
        values (torch.Tensor): critic's expected values of the policy at each state (B, E)
        dones (torch.Tensor): boolean flags indicating whether an episode is done (1) or not (0) (B, E)
        discount_factor (float): discount factor

    Returns:
        torch.Tensor: advantages (B, E)
    """
    returns = torch.zeros_like(values) # (B, E)
    # recursively compute ground-truth returns
    for t in range(rewards.size(-1) - 1, -1, -1):
        returns[..., t] = rewards[..., t] + resolve(discount_factor) * returns[..., t+1] * (1 - dones[..., t])
    # then compute the difference
    advantages = returns - values
    return advantages