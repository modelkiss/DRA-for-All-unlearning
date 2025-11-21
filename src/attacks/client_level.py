"""Client-level reconstruction attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import torch

from .common import DiffusionPrior, average_param_diffs, compute_param_diff, gradient_matching_loss


@dataclass
class ClientLevelTarget:
    target_client: int
    signal: Mapping[str, torch.Tensor]


def build_client_level_target_signal(
    *,
    snapshots_before: Sequence[Mapping[str, torch.Tensor]],
    snapshots_after: Sequence[Mapping[str, torch.Tensor]],
    target_client: int,
    client_update_log: Iterable[tuple[int, Sequence[torch.Tensor]]] | None = None,
) -> ClientLevelTarget:
    """Approximate the contribution gap left by removing a client."""

    diffs = []
    for before_state, after_state in zip(snapshots_before, snapshots_after):
        diffs.append(compute_param_diff(before_state, after_state))

    signal = average_param_diffs(diffs)
    if client_update_log:
        accumulated: MutableMapping[str, torch.Tensor] = {}
        for client_id, deltas in client_update_log:
            if client_id != target_client:
                continue
            for idx, delta in enumerate(deltas):
                key = f"client_update_{idx}"
                accumulated[key] = accumulated.get(key, torch.zeros_like(delta)) + delta
        signal = {**signal, **accumulated}
    return ClientLevelTarget(target_client=target_client, signal=signal)


def run_client_level_attack(
    model: torch.nn.Module,
    prior: DiffusionPrior,
    target: ClientLevelTarget,
    *,
    batch_size: int = 32,
    steps: int | None = None,
) -> torch.Tensor:
    """Generate representative samples for a forgotten client's distribution."""

    def guidance_fn(candidates: torch.Tensor) -> torch.Tensor:
        return gradient_matching_loss(model, candidates, target.signal)

    return prior.guided_sampling(batch_size=batch_size, guidance_fn=guidance_fn, num_steps=steps)
