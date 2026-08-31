from __future__ import annotations
from typing import Tuple, Any, Callable, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim

import copy

from policystack.utils.buffers import Replay
from policystack.utils.config import DynamicTerm
from policystack.training import ValueBasedTrainer, TrainingContext
from policystack.managers.actions import ActionManager
from policystack.math.objective import msbe
from policystack.rl.base import RLAlgorithm

from dataclasses import dataclass, field
from typing import Callable, Any

if TYPE_CHECKING:
    from managers.actions import ActionConfig


class DQN(RLAlgorithm):
    
    def __init__(self, config: DQNConfig) -> None:
        super().__init__()
        self.net = config.net
        
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Assembles and returns an action index"""
        out = self.net(obs) # (B, E, A)
        # make the action distributions
        self.action_manager.make_dists(out)
        # return a sampled action
        return self.sample_action(deterministic=deterministic)
    
    
    def q_values(self) -> torch.Tensor:
        """Assumes the correct distribution to be assembled; returns q-values as given by the network"""
        # sample using the current distribution
        return self.action_manager.logits() # (B, E, L)
    
    
    
class DQNTrainer(ValueBasedTrainer):
    
    def _pre_training(self) -> None:
        # instantiate replay buffer
        self.replay = Replay(["obs", "next_obs", "q_values", "rewards", "dones"], length=self.config.buffer_size)
        obs, _ = self.env.reset()
        self.replay.stage({"obs": obs})
        
        # collect preliminary samples
        # no policy refinement
        for i in range(self.config.warmup):
            self._collect_transitions()
        # establish target policy
        self._update_frozen_policy()
        
        
    def _pre_collection(self): self.context.clear()
        
        
    def _collect_transitions(self) -> None:
        obs = self.replay.staged["obs"]
        idx = self.algorithm(obs)
        # index q-value
        # usable q-values must not have exploration applied
        action_qval = self.algorithm.q_values()[..., idx]
        next_obs, reward, term, trunc, _ = self.env.step(idx)
        # save "prior" obs
        self.replay.stage(
            {"next_obs": next_obs, "q_values": action_qval, "rewards": reward, "dones": term | trunc}
        )
        
        self.context.write(
            obs=obs, action_idx=idx,
            training_q_values=self.algorithm.q_values(),
            rewards=reward
        )
        
        self.replay.commit()
        # restage next obs for subsequent step
        self.replay.stage({"obs": next_obs})
        
        
    def _pre_learning(self): self.context.clear()
        
        
    def _gradient_update(self, batch: dict[str, torch.Tensor]) -> None:
        # update policy
        self.config.op.zero_grad()
        # next q-value must be computed using the target
        idx = self.target_policy(batch["next_obs"])
        next_qval = self.target_policy.q_values()[..., idx]
        # compute current distributions
        self.target_policy(batch["next_obs"])
        loss = self.config.loss_fn(
            reward=batch["rewards"],
            value=batch["q_values"][..., idx],
            next_value=next_qval,
            done=batch["dones"],
            **self.config.loss_params,
        )
        loss.backward()
        
        self.context.write(grad_norm=nn.utils.clip_grad_norm_(self.algorithm.policy.parameters(), max_norm=float("inf")))
        
        self.config.op.step()
        
        self.context.write(
            training_q_values=batch["q_values"], action_idx=idx,
            target_next_q_values=next_qval,      loss=loss
        )
        
        
    def _update_frozen_policy(self) -> None:
        # update at frequency
        if self.context.num_updates % self.config.target_update_interval == 0:
            self.target_policy = copy.deepcopy(self.algorithm) # algorithm is a shell for the net argument's parameters
            # set for eval only
            self.target_policy.eval()
                
                
                
@dataclass
class DQNConfig:
    net: nn.Module
    # all raw logits pass through the action manager, then are divided into distributions specified by ActionTerms
    # and are recombined into action values, entropy, or lob probs
    action_config: ActionConfig
    
    
    
@dataclass
class DQNTrainerConfig:
    
    op: optim.Optimizer
    # environment must follow gymnasium convention
    # step(action) -> (obs, reward, term, trunc, info)
    # reset(seed=None) -> (obs, info)
    environment: object
    
    context: TrainingContext = TrainingContext(["global_step", "iteration", "num_updates", "epoch", "elapsed_time", "grad_norm", "obs", "action_idx", "rewards", "training_q_values", "target_next_q_values", "loss"])
    
    # gradient update fields
    loss_fn: Callable = msbe
    loss_params: dict[str, Any] = field(default_factory=dict)
    
    # number of warmup steps
    warmup: int = 8_000
    # number of collection + refinement cycles
    iterations: int = 100_000
    # define the ratio between environment and gradient update steps
    collection_freq: DynamicTerm | int = 1
    refinement_freq: DynamicTerm | int = 1
    
    buffer_size: DynamicTerm | int = 1_000_000
    batch_size: DynamicTerm | int = 64
    
    target_update_interval: DynamicTerm | int = 8_000 # gradient updates