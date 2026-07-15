import torch
import torch.nn as nn
import torch.nn.functional as F


def clipped_surrogate_objective(
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        clipping_parameter: float,
    ) -> torch.Tensor:
    """
    compute the policy objective at a state for PPO
    
    Args:
        log_prob (torch.Tensor): log probability of the chosen action being chosen in the current policy (B, 1)
        old_log_prob (torch.Tensor): log probability of the chosen action being chosen in the old policy (B, 1)
        advantage (torch.Tensor): benefit of the policy's choice over what was expected by the critic (B, 1)
        clipping_parameter (float): amount of deviation from the old policy (expressed as a ratio of probabilities) that is accepted
        
    Returns:
        torch.Tensor: the objective (B, 1)
    """
    # ratio between current policy and frozen policy aids in preventing excessive updates
    # logarithmic properties are used to smooth computations for extremely small probabilities
    ratio = torch.exp(
        torch.log(log_prob) - torch.log(old_log_prob)
    )
    # consider the benefits of the clipped surrogate objective in four cases
    # advantage is positive, ratio is clipped above 1 + epsilon: rate at which this good action is encouraged is limited to prevent a catastrophically large update
    # advantage is positive, ratio is clipped below 1 - epsilon: rate at which this good action is taken is limited currently and should be more common (i.e. policy can catch up)
    # advantage is negative, ratio is clipped above 1 + epsilon: rate at which this bad action is taken is excessive currently and should be less common (i.e. policy can catch up)
    # advantage is negative, ratio is clipped below 1 - epsilon: rate at which this bad action is discourage is limited to prevent a catastrophically large update
    objective = torch.min(
        advantage * ratio,
        advantage * torch.clip(
            ratio, 1 + clipping_parameter, 1 - clipping_parameter
        )
    )
    return objective
    
    
def ac_critic_loss(
    expected_value: torch.Tensor,
    old_expected_value: torch.Tensor,
    advantage: torch.Tensor,
) -> torch.Tensor:
    """
    compute the value loss at a state for AC algorithms

    Args:
        expected_value (torch.Tensor): critic expected value of the policy given the state (B, 1)
        old_expected_value (torch.Tensor): old critic expected value of the policy given the state (B, 1)
        advantage (torch.Tensor): benefit of the policy's choice over what was expected by the critic (B, 1)

    Returns:
        torch.Tensor: the loss (B, 1)
    """
    loss = F.mse_loss(
        expected_value,
        old_expected_value + advantage
    )
    return loss