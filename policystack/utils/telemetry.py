from __future__ import annotations
from typing import TYPE_CHECKING



import torch
from dataclasses import dataclass
from enum import Enum, auto



class Hook(Enum):
    BEFORE_TRAINING = auto() # once before any training steps happen
    ITERATION_START = auto() # every iteration, before an iteration logic
    # before and after each environment step (+ management)
    BEFORE_COLLECTION = auto()
    AFTER_COLLECTION = auto()
    # before and after each environment
    BEFORE_LEARNING = auto()
    AFTER_LEARNING = auto()



class TelemetryTerm:
    def __init__(self, hook: Hook, )



class TelemetryManager:
    def __init__(self, config: TelemetryConfig) -> None:
        
        
        
@dataclass
class TelemetryConfig:
    hooks: list[TelemetryHooks]