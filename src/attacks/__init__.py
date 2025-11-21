"""Attack modules for different unlearning granularities.

The subpackages expose composable utilities for reconstructing forgotten
information using gradient matching and diffusion-style guidance. Each module
is intentionally lightweight so it can be swapped or extended in follow-up
iterations.
"""

from .common import (
    DiffusionPrior,
    average_param_diffs,
    compute_param_diff,
    gradient_matching_loss,
)
from .client_level import build_client_level_target_signal, run_client_level_attack
from .class_level import build_class_level_target_signal, run_class_level_attack
from .sample_level import build_sample_level_target_signal, run_sample_level_attack

__all__ = [
    "DiffusionPrior",
    "average_param_diffs",
    "compute_param_diff",
    "gradient_matching_loss",
    "build_client_level_target_signal",
    "run_client_level_attack",
    "build_class_level_target_signal",
    "run_class_level_attack",
    "build_sample_level_target_signal",
    "run_sample_level_attack",
]
