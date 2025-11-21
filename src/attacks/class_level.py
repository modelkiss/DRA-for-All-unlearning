"""Class-level reconstruction attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .common import DiffusionPrior, average_param_diffs, compute_param_diff, gradient_matching_loss


@dataclass
class ClassLevelTarget:
    target_class: int
    signal: Mapping[str, torch.Tensor]


def build_class_level_target_signal(
    *,
    snapshots_before: Sequence[Mapping[str, torch.Tensor]],
    snapshots_after: Sequence[Mapping[str, torch.Tensor]],
    target_class: int,
) -> ClassLevelTarget:
    """Focus on the output layer changes for a forgotten class."""

    focused_diffs = []
    for before_state, after_state in zip(snapshots_before, snapshots_after):
        diff = compute_param_diff(before_state, after_state)
        class_diff = {name: tensor for name, tensor in diff.items() if "weight" in name or "bias" in name}
        focused_diffs.append(class_diff)
    signal = average_param_diffs(focused_diffs)
    return ClassLevelTarget(target_class=target_class, signal=signal)


def _class_guidance_fn(
    model: torch.nn.Module,
    candidates: torch.Tensor,
    target: ClassLevelTarget,
) -> torch.Tensor:
    outputs = model(candidates)
    if not isinstance(outputs, torch.Tensor):
        outputs = outputs[0]
    target_labels = torch.full((candidates.size(0),), target.target_class, device=candidates.device)
    class_loss = F.cross_entropy(outputs, target_labels)
    grad_loss = gradient_matching_loss(model, candidates, target.signal)
    return class_loss + grad_loss


def run_class_level_attack(
    model: torch.nn.Module,
    prior: DiffusionPrior,
    target: ClassLevelTarget,
    *,
    batch_size: int = 32,
    steps: int | None = None,
) -> torch.Tensor:
    """Generate candidate images representative of a forgotten class."""

    def guidance_fn(candidates: torch.Tensor) -> torch.Tensor:
        return _class_guidance_fn(model, candidates, target)

    return prior.guided_sampling(batch_size=batch_size, guidance_fn=guidance_fn, num_steps=steps)
