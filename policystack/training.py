from __future__ import annotations
from typing import Any, TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from time import time

from policystack.utils.buffers import Rollout, Replay
from policystack.utils.config import DynamicTerm
from policystack.rl.base import RLAlgorithm



class TrainingContext:
    """
    Captures per-step algorithm context
    """
    _context: dict[str, Any]
    
    def __init__(self, labels: list[str] = ["iteration", "global_env_steps", "iteration_env_steps", "global_update_steps", "iteration_update_steps", "reward", "total_iterations", "total_env_steps", "elapsed_time"]) -> None:
        self._labels = labels
        self.reset()
        
        
    def __repr__(self) -> str:
        return f"TrainingContext(context={self._context})"
        
    
    def __getitem__(self, key: str) -> Any:
        self._update_timer()
        # enforce key as element of labels
        if not key in self.content.keys():
            raise KeyError(f"'{key}'")
        return self._context[key]
    
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._update_timer()
        # enforce key as element of labels
        if not key in self.content.keys():
            raise KeyError(f"'{key}'")
        self._context[key] = value
        
        
    def __contains__(self, key: str):
        self._update_timer()
        # operates to determine whether a label has an associated value
        return not self._context[key] is None
    
    
    def clear(self) -> None:
        # reset is called at init and periodically in an algorithm as to avoid conflating metrics from different training stages
        self._context = {label: None for label in self._labels}
    
    
    def write(self, enforce_presence: bool = False, **ctx: Any) -> None:
        self._update_timer()
        for key, value in ctx.items():
            # intercept tensors and remove gradients automatically
            if isinstance(value, torch.Tensor):
                value = value.detach()
            # key exists => add to context
            if key in self._context.keys():
                self.key = value
            # key doesn't exist + enforce => raise
            elif enforce_presence:
                raise KeyError(f"'{key}'")
            
    
    def read(self, *labels: str, enforce_presence: bool = False) -> dict[str, Any] | Any:
        self._update_timer()
        # resolve labels
        if labels is None:
            return self._context
        # iteratively read content
        content = dict()
        for key in labels:
            # key exists => return in content
            if key in self._context.keys():
                content[key] = self._context[key]
            # key doesn't exist + enforce => raise
            elif enforce_presence:
                raise KeyError(f"'{key}'")
        
        return content.values()[0] if len(content.keys()) == 1 else content # return a singular value if appropriate
        
            
    def initialize_timer(self) -> None:
        self._context["start_time"] = time()
        self._context["elapsed_time"] = 0.0
        
    
    def _update_timer(self) -> None: self._context["elapsed_time"] = time() - self._context["start_time"]
        


"""
Trainer templates, rather than providing substantial functionality for inheriting
trainers, serve primarily as a template for constructing and easily reading algorithm
trainers, also enabling simple modification of existing trainers without rewriting full
training loops.
"""
   
        
class OnPolicyACTrainer(ABC):
    """
    Abstracted trainer for on-policy actor-critic algorithms
    
    The training loop is standardized across mainstream on-policy AC algorithms, and trainers should only need to implement abstract methods
    """
    def __init__(self, config, algorithm: RLAlgorithm) -> None:
        self.config = config
        self.algorithm = algorithm
        self.context: TrainingContext = config.context
        
        # map env for easy access
        self.env = self.config.environment
        # _pre_training() should assign the rollout as an attribute of the class
        self._pre_training()
        # verify that is done
        if not isinstance(getattr(self, "rollout", None), Rollout):
            raise ValueError(f"_pre_training() expected to assign attribute 'rollout' of type {Rollout.__name__}, not found")
        
        self.context.write(global_step=0, num_updates=0)
        
        
    @abstractmethod
    def _pre_training(self) -> None:
        """Constructs the rollout object using relevant transition values and any other necessary functions; called once before any iteration"""
        ...
        
        
    def _pre_collection(self) -> None:
        """No dedicated purpose in most AC algorithms, though available for custom trainers; called before collection phase, once per iteration"""
        ...
        
    
    @abstractmethod
    def _collect_transition(self) -> None:
        """Stages transition features from environment into rollout; called during every collection iteration, until rollout is full"""
        ...
    
    
    @abstractmethod
    def _pre_learning(self) -> None:
        """Computes advantages and any other functionality that must happen prior to policy refinement; called before learning phase, once per iteration"""
        ...
        
        
    def _pre_update(self) -> None:
        """No dedicated purpose in most AC algorithms, available for custom trainers; called once every learning epoch"""
        ...
        
        
    @abstractmethod
    def _gradient_update(self, batch: dict[str, Any]) -> None:
        """Performs gradient updates on actor/critic and other networks; called during every learning iteration, for every batch"""
        ...
        
        
    def train(self) -> None:
        # rollout collection phase
        for iteration in range(self.config.iterations):
            self.context.write(iteration=iteration)
            # call pre-collection hook
            self._pre_collection()
            while not self.rollout.full():
                self._collect_transition()
                
                self.context.write(global_step=self.context.read("global_step") + 1)
            
            self._pre_learning()
            # batch data
            dataloader = DataLoader(
                self.rollout,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=Rollout.collate
            )
            for epoch in range(self.config.epochs):
                # call pre-update hook
                self._pre_update()
                for batch in dataloader:
                    self._gradient_update(batch)
                
                    self.context.write(num_updates=self.context.num_updates + 1)
                    
                    
                    
class ValueBasedTrainer(ABC):
    
    def __init__(self, config, algorithm: RLAlgorithm) -> None:
        self.config = config
        self.algorithm = algorithm
        self.context: TrainingContext = config.context
        
        # map env for easy access
        self.env = self.config.environment
        # _pre_training() should assign the replay as an attribute of the class
        self._pre_training()
        # verify that is done
        if not isinstance(getattr(self, "replay", None), Replay):
            raise ValueError("_pre_training() expected to assign attribute 'replay' of type Replay, not found")
        
        self.context.write(
            global_step=0, 
            num_updates=0
        )
        
        
    @abstractmethod
    def _pre_training(self) -> None:
        """Constructs the rollout object using relevant transition values and any other necessary functions; called once before any iteration"""
        ...
        
        
    def _pre_collection(self) -> None:
        """No dedicated purpose in most value-based algorithms, though available for custom trainers; called before collection phase, once per iteration"""
        ...
        
        
    @abstractmethod
    def _collect_transitions(self) -> None:
        """Collects and logs transition from environment into rollout; called a specified number of times during an iteration, prior to learning"""
        ...
        
        
    def _pre_learning(self) -> None:
        """Computes advantages and any other functionality that must happen prior to policy refinement; called before learning phase, once per iteration"""
        ...
        
        
    @abstractmethod
    def _gradient_update(self, batch: dict[str, Any]) -> None:
        """Performs gradient updates on actor/critic and other networks; called a specified number of times during an iteration, after collection"""
        ...
        
        
    def _update_frozen_policy(self) -> None:
        """Conditionally update the frozen policy used to bootstrap the value objective"""
        ...
        
        
    def train(self) -> None:
        for iteration in range(self.config.iteration):
            
            self.context.write({"iteration", iteration})
            # collection phase
            self._pre_collection()
            for step in range(self.config.n_collections_per_iter):
                self._collect_transitions()
                
                self.context.write(global_step=self.context.read("global_step") + 1)
                self.context.write(fraction_complete=self.context.read("global_step") / self.context.read("total_steps"))
                
            # learning phase
            self._pre_learning()
            for step in range(self.n_refinements_per_iter):
                batch = self.replay.manual_batch(self.config.batch_size)
                self._gradient_update(batch)

                # updating frozen policy
                self._update_frozen_policy()
                
                self.context.write(num_updates=self.context.num_updates + 1)