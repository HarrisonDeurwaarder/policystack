import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader

from config import DynamicTerm, resolve
from utils.buffers import Rollout
from math.advantage import gae
from math.objective import clipped_surrogate_with_entropy, critic_mse
from training import TrainingState, OnPolicyACTrainer

from typing import Tuple, Any, Callable
from dataclasses import dataclass, field, MISSING



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



class PPOTrainer(OnPolicyACTrainer):
    """
    high-level ppo training handler
    """
    def _pre_training(self) -> None:
        # instanciate rollout
        self.rollout = Rollout(stackable=["obs", "actions", "log_probs", "rewards", "values", "next_values", "dones", "entropy"])
        
        
    def _pre_collection(self) -> None:
        obs, _ = self.env.reset()
        value = self.ppo.get_value(obs)
        self.rollout.reset()
        self.rollout.stage(fields={"obs": obs, "values": value})
        
        
    def _collect_transition(self) -> None:
        # compute and sample action
        obs = self.rollout.from_staged("obs")
        value = self.rollout.from_staged("obs")
        action, dist = self.ppo(obs)
        log_prob = dist.log_prob(action)
        # compute entropy for entropy term
        entropy = dist.entropy()
        
        next_obs, reward, term, trunc, _ = self.env.step(action)
        # compute critic value for next state
        next_value = self.ppo.get_value(next_obs)
        done = term | trunc
        # compute the next expected value for advantage comps
        # log transition
        self.rollout.stage(fields={
            "actions": action, "log_probs": log_prob, 
            "rewards": reward, "values": value, "next_values": next_value, 
            "dones": done, "entropy": entropy
        })
        # obs have already been added; by staging everything else, we now have a full transition
        self.rollout.commit()
        # stage the obs for the next cycle
        # as a result, an additional obs remains in the buffer after collection
        # this is eliminated when rollout.reset() is called
        self.rollout.stage(fields={"obs": next_obs, "values": next_value})
        
        
    def _pre_learning(self) -> None:
         # compute advantages across rollout
        advantages = self.config.advantage_fn(
            rewards=self.rollout.rewards, 
            expected_values=self.rollout.values, 
            dones=self.rollout.dones, 
            **self.config.advantage_params,
        )
        self.rollout.annotate("advantages", advantages)
        
    
    def _gradient_update(self, batch):
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
    epochs: int | DynamicTerm = 10
    # number of collect -> train cycles
    iterations: int = 200
    batch_size: int | DynamicTerm = 64
    # number of steps collected in rollout phase
    rollout_length: int | DynamicTerm = 1024
    
    # alloted ratio-difference between the target policy and trained policy
    # prevents caatstrophic poicy collapse by limited the amount the policy can learn in on cycle
    #clipping_param: float = 0.2
    
    policy_objective_fn: Callable = clipped_surrogate_with_entropy
    advantage_fn: Callable = gae
    critic_loss_fn: Callable = critic_mse
    
    policy_objective_params: dict[str, Any] = {
        "clipping_param": 0.2, "entropy_coef": 0.01
    }
    advantage_params: dict[str, Any] = {
        "discount_factor": 0.99, "gae_decay": 0.98
    }
    critic_loss_fn: dict[str, Any] = {}
    
    # enables the use of a single optimizer on a weighted sum of the policy and value objectives; use with a shared backbone
    # otherwise, two 
    #couple_objectives: bool = True
    # compresses the range of designated variance outputs to (0, inf)
    #exponentiate_variance: bool = True