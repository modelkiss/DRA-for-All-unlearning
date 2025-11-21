"""Sample-level reconstruction attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .common import DiffusionPrior, average_param_diffs, compute_param_diff, gradient_matching_loss


@dataclass
class SampleLevelTarget:
    signal: Mapping[str, torch.Tensor]
    label: int | None = None


def build_sample_level_target_signal(
    *,
    snapshots_before: Sequence[Mapping[str, torch.Tensor]],
    snapshots_after: Sequence[Mapping[str, torch.Tensor]],
) -> SampleLevelTarget:
    """Use the immediate difference around the unlearning event as guidance."""

    diffs = []
    for before_state, after_state in zip(snapshots_before, snapshots_after):
        diffs.append(compute_param_diff(before_state, after_state))
    signal = average_param_diffs(diffs)
    return SampleLevelTarget(signal=signal)


def _sample_guidance_fn(
    model: torch.nn.Module, candidates: torch.Tensor, target: SampleLevelTarget
) -> torch.Tensor:
    grad_loss = gradient_matching_loss(model, candidates, target.signal)
    if target.label is None:
        return grad_loss
    outputs = model(candidates)
    if not isinstance(outputs, torch.Tensor):
        outputs = outputs[0]
    labels = torch.full((candidates.size(0),), target.label, device=candidates.device)
    class_loss = F.cross_entropy(outputs, labels)
    return grad_loss + 0.5 * class_loss


def run_sample_level_attack(
    model: torch.nn.Module,
    prior: DiffusionPrior,
    target: SampleLevelTarget,
    *,
    batch_size: int = 32,
    steps: int | None = None,
    total_candidates: int = 128,
) -> torch.Tensor:
    """Generate and rank candidate reconstructions for a single forgotten sample."""

    all_candidates = []
    per_batch = max(batch_size, 1)
    batches = max(total_candidates // per_batch, 1)

    for _ in range(batches):
        def guidance_fn(candidates: torch.Tensor) -> torch.Tensor:
            return _sample_guidance_fn(model, candidates, target)

        samples = prior.guided_sampling(
            batch_size=per_batch, guidance_fn=guidance_fn, num_steps=steps
        )
        scores = _sample_guidance_fn(model, samples, target).detach()
        all_candidates.append((scores, samples.detach()))

    # Stack and select best-scoring candidates
    if not all_candidates:
        return torch.empty(0)
    scores = torch.cat([entry[0].flatten() for entry in all_candidates])
    images = torch.cat([entry[1] for entry in all_candidates])
    best_indices = torch.topk(-scores, k=min(4, images.size(0))).indices
    return images[best_indices]
