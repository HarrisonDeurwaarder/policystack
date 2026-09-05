from __future__ import annotations
from typing import TYPE_CHECKING, Any
from collections.abc import Callable



import torch
from dataclasses import dataclass
from enum import Enum, auto

if TYPE_CHECKING:
    from policystack.training import TrainingContext


        
def default_display_fn(metrics: dict[str, Any], label: str, hook_index: int, hook_max: int) -> None:
    width = 72
    # label header
    print("\n" + "="*width)
    print(f"{label.title()} {hook_index}/{hook_max}".center(width), end="\n")
    # metrics/content
    for metric, value in metrics.items():
        print(f"{(metric + ':').center(width)} {value}")
    # ending bar
    print("\n" + "="*width)



class Hook(Enum):
    BEFORE_TRAINING = auto() # once before any training steps happen
    ITERATION_START = auto() # every iteration, before an iteration logic
    # before and after each environment step (+ management)
    BEFORE_COLLECTION = auto()
    AFTER_COLLECTION = auto()
    # before and after each gradient update
    BEFORE_LEARNING = auto()
    AFTER_LEARNING = auto()



class TelemetryTerm:
    def __init__(
        self, 
        metrics: list[str],
        hook: int,
        label: str | None = None,
        frequency: int = 1,
        display_fn: Callable[[dict[str, Any]], None] = default_display_fn
    ):
        self.metrics = metrics
        self.hook = hook
        self.label = label
        self.frequency = frequency
        self.display_fn = display_fn
        
        
    def __call__(self) -> None:
        pass
    



class TelemetryManager:
    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config
        self.terms = config.terms
    
    
    def _seperate_metrics_by_hook(self) -> None:
        # by hook, assemble all the metrics that are used by it
        self._metrics_by_hook = {
            hook: set(*[term.metrics for term in self.terms]) # prevent duplicate metrics from being evaluated individually
            for hook in Hook
        }
        self._metric_history = {
            metric: list() for hook in self._metrics_by_hook.keys()
            for metric in self._metrics_by_hook[hook]
        }
    
    
    def trigger(self, hook: int) -> None:
        # get the subset of metrics required by terms using the given hook
        for metric in self._metrics_by_hook[hook]:
            self._metric_history[metric].append
        
        
        
@dataclass
class Metric:
    fn: Callable[[Any], Any]
    # enable use of only a subset of stored data
    window: int = float("inf")
    params: dict[str, str] | list[str]
        
        
        
@dataclass
class TelemetryConfig:
    # individual telemetry checks compiling different metrics
    # many implementations will only use one term
    # terms are evaluated in the order they are defined; for instance, if two per-epoch terms are added and the latter contains clear_terminal=True,
    # only that one will be displayed
    terms: list[TelemetryTerm]
    # all metrics that are used must be defined here, and will be evaluated when 
    metric_sheet: dict[str, Callable]
    # if a context component required by metric is None, then do/do not error
    # else, the metric will be None
    enforce_non_null_context = False