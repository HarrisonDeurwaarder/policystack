import torch
import torch.nn as nn

from config import DQNConfig
from utils.buffers import Replay

from dataclasses import dataclass, field
from typing import Callable


class DQN(nn.Module):
    
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        
    
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        super().__call__()
        
        
    def forward(self, obs: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        # extract action-value (q-values)
        q_values = self.net(obs) # (..B, N)
        # apply epsilon-greedy exploration
        idxs = torch.where(
            torch.rand(q_values.shape[:-1]) < epsilon, # (..B,)
            torch.randint(low=0, high=q_values.shape[-1], size=q_values.shape[:-1]), # (..B,)
            torch.argmax(q_values, dim=-1) # (..B,)
        )
        return idxs, torch.gather(q_values, -1, idxs)
    
    
    
class DQNTrainer:
    def __init__(self, config: DQNConfig, dqn: DQN) -> None:
        self.config = config
        self.dqn = dqn
        
        
    def train(self) -> None:
        env = self.config.environment
        replay = Replay(stackable=["obs", "next_obs", "actions", "rewards", "dones"])
        next_obs, _ = env.reset()
        
        # collect preliminary samples
        # no policy refinement
        steps = 0
        while len(replay) < self.config.warmup:
            steps += 1
            epsilon = self.config.epsilon_fn(steps)
            # sample action (in practice, this will be effectively random)
            action = self.dqn(obs, epsilon)
            # save "prior" obs
            obs = next_obs
            next_obs, reward, term, trunc, _ = env.step(action)
            replay.add(
                {"obs": obs, "next_obs": next_obs, "actions": action, "rewards": reward, "dones": term | trunc}
            )
            
        # now, begin policy refinement
        for iteration in range(self.config.iterations):
            for collection_step in range(self.n_collections_per_iter):
                # update total steps for epsilon scheduling
                steps += 1
                epsilon = self.config.epsilon_fn(steps)
                
                action = self.dqn(obs, epsilon)
                # save "prior" obs
                obs = next_obs
                obs, reward, term, trunc, _ = env.step(action)
                replay.add(
                    {"obs": obs, "next_obs": next_obs, "actions": action, "rewards": reward, "dones": term | trunc}
                )
            for refinement_step in range(self.config.n_refinements_per_iter):
                batch = replay.manual_batch(self.config.batch_size)
                self.update()
                
                
                
@dataclass
class DQNConfig:
    net: nn.Module = field(default=MISSING)
    # enables exploration in the dqn; policy with a probability of epsilon selects a random action
    epsilon_fn: Callable | float = field(
        default=lambda t: max(0.01, 1.0 - t / 100_000)
    )
    
    # number of collection + refinement cycles
    iterations: int = field(default=100_000)
    # define the ratio between environment and gradient update steps
    n_collections_per_iter: int = field(default=1)
    n_refinements_per_iter: int = field(default=1)
    
    batch_size: int = field(default=64)