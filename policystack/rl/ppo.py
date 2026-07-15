import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader

from config import PPOConfig
from utils.containers import Rollout
from math.advantage import gae
from math.objective import clipped_surrogate_objective, critic_mse

from typing import Tuple, Any


class Trainer(nn.Module):
    """
    high-level ppo training handler
    
    may also be used during inference
    """
    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = Actor(config.actor)
        self.critic = Critic(config.critic)
        
        
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return super().__call__(obs)
    
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        cont_dist = self.policy.forward(obs)
        # calling the trainer's forward will return high-level usable action values
        cont_actions = cont_dist.sample()
        # concatenate and return
        return cont_actions
    
    
    def update(self, batch: dict[str, Any]) -> None:
        # update policy
        self.config.actor_op.zero_grad()
        # compute current distributions
        log_probs = self.policy(batch["obs"]).log_prob(batch["actions"])
        act_loss = self.config.policy_objective_fn(
            log_prob=log_probs,
            old_log_prob=batch["log_probs"],
            advantage=batch["advantages"],
            entropy=batch["entropy"],
            **self.config.policy_objective_params,
        )
        act_loss.backward()
        self.config.actor_op.step()
        
        # update critic
        self.config.critic_op.zero_grad()
        # compute new expected values
        values = self.critic(batch["obs"])
        crit_loss = self.config.critic_loss_fn(
            expected_value=values,
            old_expected_value=batch["values"],
            advantage=batch["advantages"],
            **self.config.critic_loss_params,
        )
        crit_loss.backward()
        self.config.critic_op.step()
    
    
    def train(self) -> None:
        env = self.config.environment
        # instanciate rollout
        rollout = Rollout(stackable=["obs", "actions", "log_probs", "rewards", "values", "dones", "entropy"])
        for iteration in range(self.config.iterations):
            # data collection phase
            # collect an initial observation
            obs, _ = env.reset()
            # reset rollout
            rollout.reset({"obs": obs})
            
            while len(rollout) < self.config.rollout_length:
                # compute and sample action
                dist = self.policy(obs)
                # sample action from distribution
                action = dist.sample()
                log_prob = dist.log_prob(action)
                # compute entropy for entropy term
                entropy = dist.entropy()
                # derive critic's expected value
                expected_value = self.critic(obs)
                
                obs, reward, term, trunc, _ = env.step(action)
                done = term | trunc
                # log transition
                rollout.add(items={
                    "obs": obs, "actions": action, "log_probs": log_prob, 
                    "rewards": reward, "values": expected_value, "dones": done,
                    "entropy": entropy,
                })
            
            # compute advantages across rollout
            advantages = self.config.advantage_fn(
                rewards=rollout["rewards"], 
                expected_values=rollout["values"], 
                dones=rollout["dones"], 
                **self.config.advantage_params,
            )
            rollout.annotate("advantages", advantages, container="stackable")
                
            # training phase
            # batch rollout
            dataloader = DataLoader(
                dataset=rollout, 
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            for epoch in range(self.config.epochs):
                for batch in dataloader:
                    # distribute update logic
                    self.update(batch)



class Actor(nn.Module):
    """
    policy function
    """
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network
        
        
    def __call__(self, obs: torch.Tensor) -> Normal:
        return super().__call__(obs)
        
        
    def forward(self, obs: torch.Tensor) -> Normal:
        """
        predict the optimal single-state action in the form of a gaussian distribution for each continous action field (e.g. the position of a joint)
        currently, only continuous actions are supported
        """
        out = self.network(obs) # (B, N*2)
        # convert to distributions
        mean, logstd = torch.chunk(out, chunks=2, dim=-1)
        return Normal(mean, torch.exp(logstd))
            
          
            
class Critic(nn.Module):
    """
    value function
    """
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network
        
    
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return super().__call__(obs)
    
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        predict the cumulative (discounted) value of the twin policy given the current state
        """
        out = self.network(obs)
        return out # (B, 1)