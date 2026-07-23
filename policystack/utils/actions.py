import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from abc import ABC, abstractmethod
from enum import Enum, auto

from config import Config, DynamicTerm

from typing import Callable



class ActionTerm:
    
    params: list[str] # distribution parameters as they are specified as arguments, e.g. Normal(loc, scale) => ["loc", "scale"]
    fn_spec: dict[str, Callable | None]
    action_dist: D.Distribution
    effective_actions: int
    
    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.effective_actions = num_actions # exceptions in the case of categorical
        
        
    def _split(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        action_params = dict()
        # tie a deterministic slice of the output to a certain parameter
        for param, logit in zip(self.params, torch.chunk(logits, chunks=len(self.params), dim=-1)):
            fn = self.fn_spec.get(param, None)
            # if fn is specified for the parameter, apply, othwerwise bundle as-is
            if fn is None: action_params[param] = logit
            else:          action_params[param] = fn(logit)
        return action_params
        
        
    @abstractmethod
    def make_dist(self, logits: torch.Tensor) -> None:
        ...
        
    
    def entropy(self) -> torch.Tensor:
        return self.action_dist.entropy() # (B, E, A)
    
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_dist.log_prob(action) # (B, E, A)
    
    
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        # sanity check
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        # exclude sample dimension if unspecified
        if n_samples > 1:
            return self.action_dist.sample((n_samples,)).transpose(-2, -1) # (B, E, n, A)
        return self.action_dist.sample() # (B, E, A)
    
    
    def deterministic_sample(self) -> torch.Tensor:
        # method may be overriden if the deterministic sample isn't logically the mode
        return self.action_dist.mode()
        
    

class GaussianAction(ActionTerm):
    
    params = ["loc", "scale"]
    fn_spec = {
        "scale": lambda x: x.clamp(-20, 2).exp() # outputting std in logspace is more stable
    }
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        action_params = self._split(logits)
        self.action_dist = D.Normal(action_params["loc"], action_params["scale"])
    
    
    
class SquashedGaussianAction(ActionTerm):
    
    params = ["loc", "scale"]
    fn_spec = {
        "scale": lambda x: x.clamp(-20, 2).exp() # std in logspace is more stable
    }
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        action_params = self._split(logits)
        self.action_dist = D.TransformedDistribution(
            D.Normal(action_params["loc"], action_params["scale"]),
            D.TanhTransform()
        )
        
        
        
class BetaAction(ActionTerm):
    
    params = ["alpha", "beta"]
    fn_spec = {
        "alpha": lambda x: F.softplus(x) + 1.0, # softplus enforces non-negativity; +1 translates parameters into more usable spaces
        "beta": lambda x: F.softplus(x) + 1.0
    }
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        action_params = self._split(logits)
        self.action_dist = D.Beta(action_params["alpha"], action_params["beta"])
        
        
        
class BernoulliAction(ActionTerm):
    
    params = ["probs"]
    fn_spec = {
        "probs": lambda x: F.sigmoid(x)
    }
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        action_params = self._split(logits)
        self.action_dist = D.Bernoulli(action_params["probs"])
        
        
        
class CategoricalAction(ActionTerm):
    
    params = ["probs"]
    fn_spec = {
        "probs": lambda x: F.softmax(x, dim=-1)
    }
    
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.effective_actions = 1
    
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        action_params = self._split(logits)
        self.action_dist = D.Independent(
            D.Categorical(action_params["probs"]), 1
        )
    
    

class GlobalStdGaussianAction(nn.Module, ActionTerm):
    
    params = ["loc"]
    fn_spec = {}
    
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.log_std = nn.Parameter(
            torch.zeros((num_actions,))
        )
    
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        action_params = self._split(logits)
        # expand std across batches and envs
        # also enable logspace learning
        effective_std = self.log_std.exp() # (A,)
        self.action_dist = D.Normal(action_params["loc"], effective_std) # (B, E, A)
        
        
        
class CustomAction(ActionTerm):
    
    def __init__(self, num_actions: int, distribution: D.Distribution, params: dict[str], fn_spec: dict[str], *transforms: D.Transform) -> None:
        self.num_actions = num_actions
        self.distribution = distribution
        self.params = params,
        self.fn_spec = fn_spec,
        self.transforms = transforms
    
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        action_params = self._split(logits)
        # enforce arg-name aligned parameters
        try: dist = self.distribution(**action_params)
        except TypeError: raise TypeError(f"Passed distribution ({self.distribution.__class__}) encounted an unexpected distribution parameter. Please ensure all parameters match arguments in the distribution.")
        
        self.action_dist = D.TransformedDistribution(
            dist, *self.transforms
        )
        
        

class ActionManager:
    
    batch_dims: tuple[int]
    
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.action_terms = cfg.action_terms
        # some terms morph logit dimensions
        self.n_effective_actions = sum([term.effective_actions for term in self.action_terms])
        
        
    def make_dists(self, logits: torch.Tensor) -> None:
        self.batch_dims = logits.shape[:-1]
        for term in self.action_terms:
            term.make_dist(logits)
    
    
    def sample(self, n_samples: int = 1, deterministic: bool = False) -> torch.Tensor:
        actions = torch.zeros(self.batch_dims + (self.n_effective_actions,))
        i_0, i_1 = 0, 0
        for term in self.action_terms:
            i_0, i_1 = i_1, term.effective_actions
            # insert sample into correct slice
            actions[..., i_0:i_1] = term.deterministic_sample(n_samples) if deterministic else term.sample(n_samples)
        return actions
    
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_probs = torch.zeros(self.batch_dims + (self.n_effective_actions,))
        i_0, i_1 = 0, 0
        for term in self.action_terms:
            i_0, i_1 = i_1, term.effective_actions
            # insert probs into correct slice
            log_probs[..., i_0:i_1] = term.log_prob(actions[i_0:i_1])
        return log_probs
    
    
    def entropy(self) -> torch.Tensor:
        entropy = torch.zeros(self.batch_dims + (self.n_effective_actions,))
        i_0, i_1 = 0, 0
        for term in self.action_terms:
            i_0, i_1 = i_1, term.effective_actions
            # insert sample into correct slice
            entropy[..., i_0:i_1] = term.entropy()