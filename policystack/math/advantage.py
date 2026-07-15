import torch
import torch.nn as nn


def gae(
        rewards: torch.Tensor,
        expected_values: torch.Tensor,
        dones: torch.Tensor,
        discount_factor: float,
        gae_decay: float,
    ) -> torch.Tensor:
    """
    compute generalized advantage estimates

    Args:
        rewards (torch.Tensor): episode rewards of the policy (1, B)
        expected_values (torch.Tensor): critic's expected values of the policy at each state (1, B+1)
        dones (torch.Tensor): boolean flags indicating whether an episode is done (1) or not (0) (1, B)
        discount_factor (float): discount factor
        gae_decay (float): gae decay

    Returns:
        torch.Tensor: _description_
    """
    # td residual component (low variance)
    # takes the difference between the netted rewards (bootstrapped with the next state to determine value) and the critic's expected value
    td_residual = rewards + discount_factor * expected_values[..., 1:] * (1 - dones) - expected_values[..., :-1]
    
    # recursive monte carlo component (low bias)
    # computes the discounted value of a finite episode directly
    advantages = torch.zeros_like(expected_values) # an additional dimension is added to ease code complexity
    for t in range(rewards.size(-1) - 1, -1, -1): # iterate backwards from T to 0, inclusive
        advantages[..., t] = td_residual[..., t] + discount_factor * gae_decay * advantages[..., t+1]
        
    return advantages[..., :-1]