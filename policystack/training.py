from dataclasses import dataclass


@dataclass
class TrainingContext:
    """
    Communicates real-time training data to modify config/curriculum...
    """
    progress: float = 0.0
    steps: int = 0
    mean_reward: float = 0.0
    kl: float = 0.0
    entropy: float = 0.0