from __future__ import annotations
from typing import Tuple, Any, Callable, TYPE_CHECKING

from policystack.utils.config import DynamicTerm, resolve
from policystack.utils.buffers import Rollout
from policystack.math.advantage import gae
from policystack.math.objective import clipped_surrogate_with_entropy, critic_mse
from policystack.training import OnPolicyACTrainer, TrainingContext
from policystack.managers.actions import ActionManager
from policystack.rl.base import RLAlgorithm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader

from typing import Tuple, Any, Callable
from dataclasses import dataclass, field, MISSING

if TYPE_CHECKING:
    from policystack.managers.actions import ActionConfig


class PPO():
    """
    Proximal policy optimization algorithm
    """
    def __init__(self, config: PPOConfig) -> None:
        super().__init__(config)
        self.policy = config.actor
        self.value = config.critic
    
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Normal:
        out = self.policy(obs) # (B, E, logits)
        # make the action distributions
        self.from_logits(out)
        # return a sampled action
        return self.sample_action(deterministic=deterministic)
    
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        value = self.value(obs)
        return value # (B, E)



class PPOTrainer(OnPolicyACTrainer):
    """
    high-level ppo training handler
    """
    
    def _pre_training(self) -> None:
        # instantiate rollout
        self.rollout = Rollout(["obs", "actions", "log_probs", "rewards", "values", "next_values", "dones"], length=self.config.rollout_length)
        # logging
        
        
    def _pre_collection(self) -> None:
        obs, _ = self.env.reset()
        value = self.algorithm.get_value(obs)
        self.rollout.reset()
        self.rollout.stage(fields={"obs": obs, "values": value.detach()})
        self.context.clear()
        
        
    def _collect_transition(self) -> None:
        # compute and sample action
        obs = self.rollout.from_staged("obs")
        value = self.rollout.from_staged("value")
        action = self.algorithm(obs).detach() # register logits
        log_prob = self.algorithm.log_prob(action).detach()
        
        next_obs, reward, term, trunc, _ = self.env.step(action)
        # compute critic value for next state
        next_value = self.algorithm.get_value(next_obs).detach()
        done = term | trunc
        # compute the next expected value for advantage comps
        # log transition
        self.rollout.stage(fields={
            "actions": action, "log_probs": log_prob, 
            "rewards": reward, "next_values": next_value.detach(), 
            "dones": done
        })
        
        self.context.write(
            obs=obs,        actions=action,
            rewards=reward, values=value,
            dones=done,     target_log_probs=log_prob
        )
        
        # obs have already been added; by staging everything else, we now have a full transition
        self.rollout.commit()
        # stage the obs for the next cycle
        # as a result, an additional obs remains in the buffer after collection
        # this is eliminated when rollout.reset() is called
        self.rollout.stage(fields={"obs": next_obs, "values": next_value.detach()})
        
        
    def _pre_learning(self) -> None:
         # compute advantages across rollout
        advantages = self.config.advantage_fn(
            rewards=self.rollout.rewards, 
            values=self.rollout.values,
            next_values=self.rollout.next_values,
            dones=self.rollout.dones, 
            **self.config.advantage_params,
        )
        self.rollout.annotate("advantages", advantages.detach())
        
        self.context.clear()
        
    
    def _gradient_update(self, batch):
        # update policy
        self.config.actor_op.zero_grad()
        # compute current distributions
        _ = self.algorithm(batch["obs"])
        entropy = self.algorithm.entropy()
        log_probs = self.algorithm.log_prob(batch["actions"])
        act_loss = self.config.policy_objective_fn(
            log_prob=log_probs,
            old_log_prob=batch["log_probs"],
            advantage=batch["advantages"],
            entropy=entropy,
            **self.config.policy_objective_params,
        )
        act_loss.backward()
        
        self.context.write(
            grad_norm=nn.utils.clip_grad_norm_(self.algorithm.policy.parameters(), max_norm=float("inf"))
        )
        
        self.config.actor_op.step()
        
        # update critic
        self.config.critic_op.zero_grad()
        # compute new expected values
        values = self.algorithm.get_value(batch["obs"])
        crit_loss = self.config.critic_loss_fn(
            expected_value=values,
            old_expected_value=batch["values"],
            advantage=batch["advantages"],
            **self.config.critic_loss_params,
        )
        crit_loss.backward()
        self.config.critic_op.step()
        
        self.context.write(
            advantages=batch["advantages"], actions=batch["actions"],
            entropy=entropy,                target_log_probs=batch["log_probs"],
            training_log_probs=log_probs,   target_values=batch["values"],
            training_values=values,         policy_loss=act_loss,
            value_loss=crit_loss
        )


@dataclass
class PPOConfig:
    # network architecture parameters
    # for the state-action function, the return value of forward() should be
    # keyed using the aliases defined in your ActionManager object
    # else, all actions will be assumed to be continuous and may impose downstream errors
    actor: nn.Module
    critic: nn.Module
    # all raw logits pass through the action manager, then are divided into distributions specified by ActionTerms
    # and are recombined into action values, entropy, or lob probs
    action_config: ActionConfig



@dataclass
class PPOTrainerConfig:
    # assumes that actor/critic are trained separately
    # i.e. no shared backbone
    actor_op: optim.Optimizer # note that learning rate scheduling is done within the optimizers; other curriculum
    critic_op: optim.Optimizer
    
    # environment must follow gymnasium convention
    # step(action) -> (obs, reward, term, trunc, info)
    # reset(seed=None) -> (obs, info)
    environment: object
    
    context: TrainingContext = TrainingContext([
        "global_step", "iteration", "num_updates", "epoch", "elapsed_time", 
        "obs", "actions", "rewards", "dones", 
        "entropy", "target_log_probs", "training_log_prob", "target_values", 
        "training_values", "grad_norm", "policy_loss", "value_loss", 
        "advantages"
    ])
    
    # number of times transitions from each rollout are iterated over
    epochs: int | DynamicTerm = 10
    # number of collect -> train cycles
    iterations: int = 200
    batch_size: int | DynamicTerm = 64
    # number of steps collected in rollout phase
    rollout_length: int | DynamicTerm = 1024
    
    # alloted ratio-difference between the target policy and trained policy
    # prevents catastrophic policy collapse by limited the amount the policy can learn in on cycle
    #clipping_param: float = 0.2
    
    policy_objective_fn: Callable = clipped_surrogate_with_entropy
    advantage_fn: Callable = gae
    critic_loss_fn: Callable = critic_mse
    
    policy_objective_params: dict[str, Any] = field(default_factory=lambda: {
        "clipping_param": 0.2, "entropy_coef": 0.01
    })
    advantage_params: dict[str, Any] = field(default_factory=lambda: {
        "discount_factor": 0.99, "gae_decay": 0.98
    })
    critic_loss_params: dict[str, Any] = field(default_factory=dict)
    
    # enables the use of a single optimizer on a weighted sum of the policy and value objectives; use with a shared backbone
    # otherwise, two 
    #couple_objectives: bool = True
    # compresses the range of designated variance outputs to (0, inf)
    #exponentiate_variance: bool = True