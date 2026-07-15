import torch
import torch.nn as nn


def gae(
    rewards: torch.Tensor,
    expected_values: torch.Tensor,
    dones: torch.Tensor,
    discount_factor: float,
    gae_decay: float,
    **kwargs
) -> torch.Tensor:
    """
    compute generalized advantage estimates

    Args:
        rewards (torch.Tensor): episode rewards of the policy (B, 1)
        expected_values (torch.Tensor): critic's expected values of the policy at each state (B+1, 1)
        dones (torch.Tensor): boolean flags indicating whether an episode is done (1) or not (0) (B, 1)
        discount_factor (float): discount factor
        gae_decay (float): gae decay

    Returns:
        torch.Tensor: advantage estimates
    """
    # td residual component (low variance)
    # takes the difference between the netted rewards (bootstrapped with the next state to determine value) and the critic's expected value
    td_res = td_residual(rewards, expected_values, dones, discount_factor)
    
    # recursive monte carlo component (low bias)
    # computes the discounted value of a finite episode directly
    advantages = torch.zeros_like(expected_values) # an additional dimension is added to ease code complexity
    for t in range(rewards.size(-1) - 1, -1, -1): # iterate backwards from T to 0, inclusive
        advantages[..., t] = td_res[..., t] + discount_factor * gae_decay * advantages[..., t+1]
        
    return advantages[..., :-1]


def td_residual(
    rewards: torch.Tensor,
    expected_values: torch.Tensor,
    dones: torch.Tensor,
    discount_factor: float,
    **kwargs
) -> torch.Tensor:
    """
    Computes the TD residual advantages
    
    TD residual is a low-variance method of advantage computation that compares the critic's expected value of
    the policy to a bootstrapped value in which the "ground truth" value is estimated using the reward and the
    critic's succeeding value estimation
    TD residual is, however, high-biased because of its reliance on the policy's own predictions to serve as the ground-truth

    Args:
        rewards (torch.Tensor): episode rewards of the policy (B, 1)
        expected_values (torch.Tensor): critic's expected values of the policy at each state (B+1, 1)
        dones (torch.Tensor): boolean flags indicating whether an episode is done (1) or not (0) (B, 1)
        discount_factor (float): discount factor

    Returns:
        torch.Tensor: advantage estimates
    """
    advantages = rewards + discount_factor * expected_values[..., 1:] (1 - dones) - expected_values[..., :-1]
    return advantages


def monte_carlo(
    rewards: torch.Tensor,
    expected_values: torch.Tensor,
    dones: torch.Tensor,
    discount_factor: float,
    **kwargs
) -> torch.Tensor:
    """
    Computes the monte carlo advantages
    
    Monte carlo advantages are low-bias that compare the critic's expectation to the true value of a state given the policy
    Though, monte carlo advantages are high-variance and do not recieve any smoothing as GAE or TD residuals would,
    resulting in noisy advantage signals

    Args:
        rewards (torch.Tensor): episode rewards of the policy (B, 1)
        expected_values (torch.Tensor): critic's expected values of the policy at each state (B+1, 1)
        dones (torch.Tensor): boolean flags indicating whether an episode is done (1) or not (0) (B, 1)
        discount_factor (float): discount factor

    Returns:
        torch.Tensor: advantages
    """
    returns = torch.zeros_like(expected_values) # (B+1, 1)
    # recursively compute ground-truth returns
    for t in range(rewards.size(-1) - 1, -1, -1):
        returns[..., t] = rewards[..., t] + discount_factor * returns[..., t+1] * (1 - dones[..., t])
    # then compute the difference
    advantages = returns[..., :-1] - expected_values
    return advantages