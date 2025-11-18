from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PartitionStrategy(str, Enum):
    DIRICHLET_NO_REPLACEMENT = "dirichlet_no_replacement"
    DIRICHLET_WITH_REPLACEMENT = "dirichlet_with_replacement"
    IID_NO_REPLACEMENT = "iid_no_replacement"


class TrainingStopMode(str, Enum):
    TARGET_ACCURACY = "target_accuracy"
    FIXED_ROUNDS = "fixed_rounds"


@dataclass
class LocalDPConfig:
    enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.1


@dataclass
class FederatedConfig:
    dataset: str
    num_clients: int = 10
    local_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    partition_strategy: PartitionStrategy = PartitionStrategy.DIRICHLET_NO_REPLACEMENT
    alpha: float = 0.5
    training_stop_mode: TrainingStopMode = TrainingStopMode.FIXED_ROUNDS
    max_rounds: int = 20
    target_accuracy: float = 0.8
    dp: LocalDPConfig = field(default_factory=LocalDPConfig)
    model_save_dir: Path = Path("artifacts/module_a_models")
    trajectory_dir: Path = Path("artifacts/trajectories")
    device: str = "cuda"


@dataclass
class ModuleDSplit:
    diffusion_indices: list
    federated_indices: list
