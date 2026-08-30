import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field, MISSING

from policystack.utils.config import DynamicTerm, resolve

from typing import Callable


class ActionTerm(ABC):
    """Abstract term defining a distribution and a set number of action dimensions; exposes sample, log_prob, and entropy distribution properties"""
    # distribution specs
    param_names: list[str] # distribution parameters as they are specified as arguments, e.g. Normal(loc, scale) => ["loc", "scale"]
    fn_spec: dict[str, Callable | None] # transformation applied to logits, by classifier
    action_dist: D.Distribution # root distribution object, independent of any pre or post transforms
    n_actions: int # number of action outputs
    n_logits: int # number of logit inputs; must be separate because, for example, a gaussian would take 2 parameters (loc, scale) per sampled scalar
    # book-keeping and distribution memory
    raw_logits: torch.Tensor # pre-transform logits
    action_params: dict[str, torch.Tensor] # distribution parameters; post transformation and split
    
    def __init__(self, n_actions: int | None = None, n_logits: int | None = None) -> None:
        # hit error cases prior to computing
        # logit and actions don't align in dimensionality
        if (not n_actions is None) and (not n_logits is None) and (n_actions * len(self.param_names) != n_logits):
            raise ValueError(f"Expected n_logits = {n_actions}*{len(self.param_names)} = {n_actions * len(self.param_names)} (n_actions * logits_per_action), got n_logits = {n_logits}. Please reconcile count or specify one.")
        # neither logits nor actions were passed
        elif (n_actions is None) and (n_logits is None):
            raise ValueError(f"Expected parameters n_actions OR n_logits to be passed, got neither.")
        # logits are indivisible by requirement
        elif (not n_actions is None) and (not n_logits is None) and (n_logits % len(self.param_names) != 0):
            raise ValueError(f"Expected n_logits divisible by logits_per_action ({len(self.param_names)}), got {n_logits}")
        # if a term uses a constant action dimension, __init__() must be overriden; see CategoricalTerm for example
        
        # else infer values
        # n_logits is specified
        if (n_actions is None) and (not n_logits is None):
            n_actions = n_logits // len(self.param_names)
        # n_actions is specified
        if (n_logits is None) and (not n_actions is None):
            n_logits = n_actions * len(self.param_names)
        
        self.n_actions = n_actions
        self.n_logits = n_logits
        
        
    def _split(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        # intercept bad logits
        if logits.shape[-1] != self.n_logits:
            raise ValueError(f"Expected logit dimension torch.Size([..., {self.n_logits}]), got {logits.shape}")
        self.raw_logits = logits
        action_params = dict()
        # tie a deterministic slice of the output to a certain parameter
        for param, logit in zip(self.param_names, torch.chunk(logits, chunks=len(self.param_names), dim=-1)):
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
            sample = self.action_dist.sample((n_samples,)).movedim(0, -2) # (B, E, n, A)
            # log an arbitrary sample 
            self.latest_sample = sample[..., 0, :] # (B, E, A)
        else:
            sample = self.action_dist.sample() # (B, E, A)
            self.latest_sample = sample[:]
        # debug check that the sampled dimensionality is as expected
        if sample.shape[-1] != self.n_actions:
            raise ValueError(f"Bad sample intercepted; expected sample dimension torch.Size([..., {self.n_actions}]), got {sample.shape}")
        return sample
    
    
    def deterministic_sample(self) -> torch.Tensor:
        # method must be overriden if the deterministic sample isn't logically the mode
        return self.action_dist.mode
    
    
    def parameters(self) -> torch.Tensor:
        """Get the distribution parameters (logits with transformations applied)"""
        return torch.stack(self.action_params.values())
        
        
    def logits(self) -> torch.Tensor:
        return self.raw_logits # (B, E, L)
        
    

class GaussianAction(ActionTerm):
    """ActionTerm sampling over a gaussian (normal) distribution; default in continuous action spaces"""
    param_names = ["loc", "scale"]
    fn_spec = {
        "scale": lambda x: x.clamp(-20, 2).exp() # outputting std in logspace is more stable
    }
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        self.action_params = self._split(logits)
        self.action_dist = D.Normal(self.action_params["loc"], self.action_params["scale"])
    
    
    
class SquashedGaussianAction(ActionTerm):
    """ActionTerm sampling over a tanh-transformed gaussian; used in SAC"""
    param_names = ["loc", "scale"]
    fn_spec = {
        "scale": lambda x: x.clamp(-20, 2).exp() # std in logspace is more stable
    }
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        self.action_params = self._split(logits)
        self.action_dist = D.TransformedDistribution(
            D.Normal(self.action_params["loc"], self.action_params["scale"]),
            D.TanhTransform()
        )
        
    def entropy(self) -> torch.Tensor:
        # no closed form exists; an estimate based on the latest computed sample is instead used
        return -self.log_prob(self.latest_sample) # (B, E, A)
    
    
    def deterministic_sample(self) -> torch.Tensor:
        # same story
        return torch.tanh(self.action_dist.base_dist.loc)
        
        
class BetaAction(ActionTerm):
    """ActionTerm sampling over a beta distribution defined by the beta function; alternative in SAC"""
    param_names = ["alpha", "beta"]
    fn_spec = {
        "alpha": lambda x: F.softplus(x) + 1.0, # softplus enforces non-negativity; +1 translates parameters into more usable spaces
        "beta": lambda x: F.softplus(x) + 1.0
    }
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        self.action_params = self._split(logits)
        self.action_dist = D.Beta(self.action_params["alpha"], self.action_params["beta"])
        
        
        
class BernoulliAction(ActionTerm):
    """ActionTerm sampling over a bernoulli distribution; e.g. open/closed claw"""
    param_names = ["probs"]
    fn_spec = {
        "probs": lambda x: F.sigmoid(x)
    }
    
    def __init__(self, n_actions: int | None = None, n_logits: int | None = None, epsilon: float | DynamicTerm = 0.0) -> None:
        super().__init__(n_actions=n_actions, n_logits=n_logits)
        self.epsilon = epsilon
    
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        self.action_params = self._split(logits)
        # apply epsilon-greedy probability
        epsilon = resolve(self.epsilon)
        probs = (1.0 - epsilon) * self.action_params["probs"] + epsilon / 2.0 # by the law of total probability
        self.action_dist = D.Bernoulli(probs)
        
        
        
class CategoricalAction(ActionTerm):
    """ActionTerm sampling over a categorical distribution (outputs only one action per term, unlike other ActionTerms); e.g. WASD"""
    param_names = ["probs"]
    fn_spec = {
        "probs": lambda x: F.softmax(x, dim=-1).unsqueeze(-2)
    }
    
    def __init__(self, n_logits: int, epsilon: float | DynamicTerm = 0.0) -> None:
        self.n_logits = n_logits
        self.n_actions = 1
        self.epsilon = epsilon
    
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        self.action_params = self._split(logits)
        # apply epsilon-greedy probability
        epsilon = resolve(self.epsilon)
        probs = (1.0 - epsilon) * self.action_params["probs"] + epsilon / self.n_logits
        self.action_dist = D.Categorical(probs)
        
        
class CategoricalDeltaAction(CategoricalAction):
    """ActionTerm sampling over a greedy categorical; used in DQNs with a nonzero epsilon for exploration"""
    def __init__(self, n_logits: int, epsilon: float | DynamicTerm = 0.0) -> None:
        super().__init__(n_logits, epsilon)
        # modify fn_spec to apply onehot
        self.fn_spec = {
            "probs": lambda x: F.one_hot(x.argmax(-1, keepdims=True), n_logits)
        }
        
    

class GlobalStdGaussianAction(nn.Module, ActionTerm):
    """ActionTerm sampling over a gaussian distribution with a universal, learned std"""
    param_names = ["loc"]
    fn_spec = {}
    
    def __init__(self, n_actions: int | None = None, n_logits: int | None = None) -> None:
        nn.Module.__init__(self)
        ActionTerm.__init__(self, n_actions=n_actions, n_logits=n_logits)
        
        self.log_std = nn.Parameter(
            torch.zeros((self.n_actions,))
        )
    
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        self.action_params = self._split(logits)
        # expand std across batches and envs
        # also enable logspace learning
        std = self.log_std.exp() # (A,)
        self.action_dist = D.Normal(self.action_params["loc"], std) # (B, E, A)
        
        
        
class CustomAction(ActionTerm):
    """ActionTerm with flexible distribution and parameter usage; if a CustomAction term is insufficient, consider inheriting from ActionTerm to define your own"""
    def __init__(self, distribution: D.Distribution, params: dict[str], fn_spec: dict[str], *transforms: D.Transform, n_actions: int | None = None, n_logits: int | None = None, ) -> None:
        super().__init__(n_actions=n_actions, n_logits=n_logits)
        self.distribution = distribution
        self.params = params
        self.fn_spec = fn_spec
        self.transforms = transforms
    
    
    def make_dist(self, logits: torch.Tensor) -> None:
        # process raw logits, splitting and applying transforms
        self.action_params = self._split(logits)
        # enforce arg-name aligned parameters
        try: dist = self.distribution(**self.action_params)
        except TypeError: raise TypeError(f"Passed distribution ({self.distribution.__class__}) encountered an unexpected distribution parameter. Please ensure all parameters match arguments in the distribution.")
        
        self.action_dist = D.TransformedDistribution(
            dist, *self.transforms
        )
        
        

class ActionManager:
    """Intercepts raw logits and supplies interpretable actions to the environment"""
    batch_dims: tuple[int]
    
    def __init__(self, config: ActionConfig) -> None:
        self.config = config
        self.action_terms: list[ActionTerm] = config.terms
        # aggregate quantities
        self.n_logits = sum([term.n_logits for term in self.action_terms])
        self.n_actions = sum([term.n_actions for term in self.action_terms])
        
        
    def make_dists(self, logits: torch.Tensor) -> None:
        # intercept bad logits
        if logits.shape[-1] != self.n_logits:
            raise ValueError(f"Expected logit dimension torch.Size([..., {self.n_logits}]), got {logits.shape}")
        self.batch_dims = logits.shape[:-1]
        i_0, i_1 = 0, 0
        for term in self.action_terms:
            i_0, i_1 = i_1, i_1 + term.n_logits
            term.make_dist(logits[..., i_0:i_1])
    
    
    def sample(self, n_samples: int = 1, deterministic: bool = False) -> torch.Tensor:
        actions = torch.zeros(self.batch_dims + (self.n_actions,)) if n_samples == 1 else torch.zeros(self.batch_dims + (n_samples, self.n_actions))
        i_0, i_1 = 0, 0
        for term in self.action_terms:
            i_0, i_1 = i_1, i_1 + term.n_actions
            # insert sample into correct slice
            actions[..., i_0:i_1] = term.deterministic_sample() if deterministic else term.sample(n_samples)
        return actions
    
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_probs = torch.zeros(self.batch_dims + (self.n_actions,))
        i_0, i_1 = 0, 0
        for term in self.action_terms:
            i_0, i_1 = i_1, i_1 + term.n_actions
            # insert probs into correct slice
            log_probs[..., i_0:i_1] = term.log_prob(actions[..., i_0:i_1])
        return log_probs
    
    
    def entropy(self) -> torch.Tensor:
        entropy = torch.zeros(self.batch_dims + (self.n_actions,))
        i_0, i_1 = 0, 0
        for term in self.action_terms:
            i_0, i_1 = i_1, i_1 + term.n_actions
            # insert sample into correct slice
            entropy[..., i_0:i_1] = term.entropy()
        return entropy
    
    
    def logits(self) -> torch.Tensor:
        logits = torch.zeros(self.batch_dims + (self.n_logits,))
        i_0, i_1 = 0, 0
        for term in self.action_terms:
            i_0, i_1 = i_1, i_1 + term.n_logits
            # insert sample into correct slice
            logits[..., i_0:i_1] = term.logits()
        return logits
    
    

@dataclass
class ActionConfig:
    terms: list[ActionTerm] = field(default=MISSING)