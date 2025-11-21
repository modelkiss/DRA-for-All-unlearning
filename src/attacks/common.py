"""Shared utilities for gradient-based reconstruction attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ParameterState = Mapping[str, torch.Tensor]


def compute_param_diff(before: ParameterState, after: ParameterState) -> Dict[str, torch.Tensor]:
    """Compute a simple parameter difference dictionary.

    The function aligns keys between ``before`` and ``after`` and only keeps
    entries present in both. This keeps the resulting signal portable even when
    models have auxiliary buffers that might differ across checkpoints.
    """

    diff: Dict[str, torch.Tensor] = {}
    for name, before_tensor in before.items():
        if name not in after:
            continue
        diff[name] = after[name] - before_tensor
    return diff


def average_param_diffs(diffs: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average a list of parameter difference dictionaries."""

    accumulated: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for diff in diffs:
        for name, tensor in diff.items():
            accumulated[name] = accumulated.get(name, torch.zeros_like(tensor)) + tensor
            counts[name] = counts.get(name, 0) + 1
    for name, tensor in accumulated.items():
        accumulated[name] = tensor / max(counts.get(name, 1), 1)
    return accumulated


@dataclass
class DiffusionPrior:
    """Lightweight diffusion-style prior used for guided sampling.

    This is a placeholder that mimics a latent diffusion loop without requiring
    a heavy pretrained checkpoint. Sampling operates directly in pixel space and
    can be swapped out with an actual diffusion model later.
    """

    image_shape: Sequence[int]
    steps: int = 30
    step_size: float = 0.1
    device: str = "cpu"

    def sample_latent(self, batch_size: int) -> torch.Tensor:
        return torch.randn((batch_size, *self.image_shape), device=self.device)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent)

    def guided_sampling(
        self,
        batch_size: int,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Run a minimal gradient-descent-like guided sampling loop."""

        steps = num_steps or self.steps
        latent = self.sample_latent(batch_size).requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=self.step_size)
        for _ in range(steps):
            optimizer.zero_grad()
            images = self.decode(latent)
            loss = guidance_fn(images)
            loss.backward()
            optimizer.step()
        return self.decode(latent).detach()


def _collect_gradients(model: nn.Module) -> Dict[str, torch.Tensor]:
    gradients: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone()
    return gradients


def gradient_matching_loss(
    model: nn.Module,
    candidate_images: torch.Tensor,
    target_signal: Mapping[str, torch.Tensor],
    loss_type: str = "mse",
) -> torch.Tensor:
    """Compute a gradient matching loss between a candidate batch and a target signal."""

    model.zero_grad(set_to_none=True)
    outputs = model(candidate_images)
    if not isinstance(outputs, torch.Tensor):
        # Fallback for models returning tuples
        outputs = outputs[0]
    pseudo_labels = outputs.argmax(dim=1)
    loss = F.cross_entropy(outputs, pseudo_labels)
    loss.backward(retain_graph=True)
    gradients = _collect_gradients(model)

    matching_terms = []
    for name, target_grad in target_signal.items():
        if name not in gradients:
            continue
        current_grad = gradients[name]
        if loss_type == "cosine":
            matching = 1 - F.cosine_similarity(current_grad.flatten(), target_grad.flatten(), dim=0)
        else:
            matching = F.mse_loss(current_grad, target_grad)
        matching_terms.append(matching)

    if not matching_terms:
        return torch.tensor(0.0, device=candidate_images.device)
    return torch.stack(matching_terms).mean()
