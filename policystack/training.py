import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.data import DataLoader

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from utils.buffers import Rollout, Replay
from config import DynamicTerm

from __future__ import annotations
from typing import Any, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from config import Config
    


@dataclass
class TrainingState:
    """
    Communicates real-time training data to modify config/curriculum
    
    DynamicTerms base curriculum updates off of TrainingState contents; all other telemetry depends on TrainingState to communicate
    between the training loop and external helpers
    TrainingState does not communicate internally during training; refer to StepContext
    """
    progress: float = 0.0
    steps: int = 0
    mean_reward: float = 0.0
    kl: float = 0.0
    entropy: float = 0.0
    
    iteration: int = 0
    epoch: int = 0
    collection_step: int = 0
    learning_step: int = 0
    
    extra: dict[str, Any] = field(default_factory=dict)
    
    
    def make_term(self, callback: Callable) -> DynamicTerm:
        # attribute new term to TrainingState object
        return DynamicTerm(callback, self)
        
        
        
class OnPolicyACTrainer(ABC):
    """
    Abstracted trainer for on-policy actor-critic algorithms
    
    The training loop is standardized across mainstream on-policy AC algorithms, and trainers should only need to implement abstract methods
    """
    def __init__(self, config: Config, algorithm: nn.Module, state: TrainingState) -> None:
        self.config = config
        self.algorithm = algorithm
        self.state = state
        
        
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
        # map env for easy access
        self.env = self.config.environment
        # _pre_training() should assign the rollout as an attribute of the class
        self._pre_training()
        # verify that is done
        if not isinstance(getattr(self, "rollout", None), Rollout):
            raise ValueError("_pre_training() expected to assign attribute 'rollout' of type Rollout, not found")
        
        # rollout collection phase
        for iteration in range(self.config.iterations):
            # call pre-collection hook
            self._pre_collection()
            while not self.rollout.full():
                self._collect_transition()
                self.rollout.commit()
            
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
                    
                    
                    
class ValueBasedTrainer(ABC):
    def __init__(self, config: Config, algorithm: nn.Module, state: TrainingState) -> None:
        self.config = config
        self.algorithm = algorithm
        self.state = state
        
        
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
        
        
    def train(self) -> None:
        # map env for easy access
        self.env = self.config.environment
        # _pre_training() should assign the replay as an attribute of the class
        self._pre_training()
        # verify that is done
        if not isinstance(getattr(self, "replay", None), Replay):
            raise ValueError("_pre_training() expected to assign attribute 'replay' of type Replay, not found")
        
        for iteration in range(self.config.iteration):
            # collection phase
            self._pre_collection()
            for i in range(self.config.n_collections_per_iter):
                self._collect_transitions()
                
            # learning phase
            self._pre_learning()
            for i in range(self.n_refinements_per_iter):
                batch = self.replay.manual_batch(self.config.batch_size)
                self._gradient_update(batch)