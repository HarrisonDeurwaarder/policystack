from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("policystack")
except PackageNotFoundError:
    __version__ = "unknown"

from policystack.training import TrainingState, OnPolicyACTrainer, ValueBasedTrainer
from policystack.utils import DynamicTerm, resolve, Rollout, Replay
from policystack.managers import ActionManager, ActionTerm
from policystack.rl import PPO, PPOTrainer, PPOConfig, PPOTrainerConfig, DQN, DQNTrainer, DQNConfig

__all__ = [
    "__version__",
    "TrainingState", "OnPolicyACTrainer", "ValueBasedTrainer",
    "DynamicTerm", "resolve", "Rollout", "Replay",
    "ActionManager", "ActionTerm",
    "PPO", "PPOTrainer", "PPOConfig", "PPOTrainerConfig",
    "DQN", "DQNTrainer", "DQNConfig",
]