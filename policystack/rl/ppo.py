import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader

from config import PPOConfig
from policystack.utils.buffers import Rollout
from math.advantage import gae
from math.objective import clipped_surrogate_objective, critic_mse

from typing import Tuple, Any



class PPO(nn.Module):
    """
    Proximal policy optimization algorithm
    """
    def __init__(self, actor: nn.Module, critic: nn.Module) -> None:
        super().__init__()
        self.policy = actor
        self.value = critic
        
    
    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        # __call__ provides both the given distribution and a sampled action
        dist = super().__call__()
        action = dist.sample()
        return action, dist
    
    
    def forward(self, obs: torch.Tensor) -> Normal:
        out = self.actor(obs) # (..B, N*2)
        # extract gaussian parameters
        mean, logvar = torch.chunk(out, chunks=2, dim=-1) # (..B, N, 2)
        # assume network to output variance in logspace
        dist = Normal(mean, torch.exp(logvar))
        return dist
    
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        value = self.value(obs)
        return value # (..B, 1)



class PPOTrainer():
    """
    high-level ppo training handler
    """
    def __init__(self, config: PPOConfig, ppo: PPO) -> None:
        super().__init__()
        self.config = config
        self.ppo = ppo
    
    
    def update(self, batch: dict[str, Any]) -> None:
        """perform a single-batch gradient update on the actor/critic"""
        # update policy
        self.config.actor_op.zero_grad()
        # compute current distributions
        _, dist = self.ppo(batch["obs"])
        log_probs = dist.log_prob(batch["actions"])
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
        values = self.ppo.get_value(batch["obs"])
        crit_loss = self.config.critic_loss_fn(
            expected_value=values,
            old_expected_value=batch["values"],
            advantage=batch["advantages"],
            **self.config.critic_loss_params,
        )
        crit_loss.backward()
        self.config.critic_op.step()
    
    
    def train(self) -> None:
        """train the policy using a given number of iterations"""
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
                action, dist = self.ppo(obs)
                log_prob = dist.log_prob(action)
                # compute entropy for entropy term
                entropy = dist.entropy()
                # derive critic's expected value
                expected_value = self.ppo.get_value(obs)
                
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