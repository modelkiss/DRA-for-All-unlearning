"""Unified experiment orchestrator for federated unlearning attacks.

The module follows the blueprint provided by the "final attack plan" and
introduces structured configuration objects plus orchestration helpers. While
individual attack implementations are lightweight placeholders, the scaffolding
mirrors the intended workflow so researchers can plug in more sophisticated
components later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from federated.config import FederatedConfig
from federated.pipeline import UNLEARNING_ALGORITHMS, module_b_federated_unlearning
from federated.training import run_module_a
from federated.models import build_model

from attacks import (
    DiffusionPrior,
    build_class_level_target_signal,
    build_client_level_target_signal,
    build_sample_level_target_signal,
    run_class_level_attack,
    run_client_level_attack,
    run_sample_level_attack,
)


DATASETS = ["CIFAR10", "CIFAR100", "FEMNIST", "SVHN"]
GRANULARITIES = ["client", "class", "sample"]
ALGORITHMS = {
    "client": ["FedEraser", "FedRecovery"],
    "class": ["FUCRT", "FUCP"],
    "sample": ["FedAF", "FedAU"],
}


@dataclass
class SnapshotConfig:
    pre_rounds: int = 3
    post_rounds: int = 3


@dataclass
class AttackHyperParams:
    diffusion_steps: int = 30
    attack_batch_size: int = 32
    total_candidates: int = 128


@dataclass
class ExperimentConfig:
    dataset: str
    granularity: str
    algorithm: str
    target_id: int = 0
    base_fed_config: FederatedConfig | None = None
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    attack: AttackHyperParams = field(default_factory=AttackHyperParams)
    device: str = "cuda"

    def resolve_federated(self) -> FederatedConfig:
        if self.base_fed_config:
            return self.base_fed_config
        return FederatedConfig(dataset=self.dataset.lower())


@dataclass
class TrainingLog:
    global_models: List[Mapping[str, torch.Tensor]]
    client_updates: Iterable


@dataclass
class UnlearningLog:
    global_models_after: List[Mapping[str, torch.Tensor]]
    metadata: Dict


@dataclass
class AttackResult:
    images: torch.Tensor
    metadata: Dict


@dataclass
class ExperimentRun:
    config: ExperimentConfig
    training_log: TrainingLog
    unlearning_log: UnlearningLog
    attack_result: AttackResult


def _load_snapshots(paths: Sequence[str]) -> List[Mapping[str, torch.Tensor]]:
    return [torch.load(path, map_location="cpu") for path in paths]


def federated_train(config: ExperimentConfig) -> TrainingLog:
    fed_cfg = config.resolve_federated()
    module_a_output = run_module_a(fed_cfg)
    snapshots = module_a_output.get("saved_models", [])[-config.snapshot.pre_rounds :]
    return TrainingLog(global_models=_load_snapshots(snapshots), client_updates=[])


def apply_unlearning(config: ExperimentConfig) -> UnlearningLog:
    fed_cfg = config.resolve_federated()
    result = module_b_federated_unlearning(
        fed_cfg,
        granularity=config.granularity,
        algorithm=config.algorithm,
        target_id=config.target_id,
        aggregation_rounds=max(config.snapshot.post_rounds, 5),
    )
    checkpoints = result.outputs.get("aggregation_checkpoints", [])
    unlearned = result.outputs.get("unlearned_checkpoint")
    combined = [unlearned] + checkpoints if unlearned else checkpoints
    snapshots = _load_snapshots(combined)[: config.snapshot.post_rounds]
    return UnlearningLog(global_models_after=snapshots, metadata=result.outputs)


def _build_prior(model: torch.nn.Module, device: str, attack_cfg: AttackHyperParams) -> DiffusionPrior:
    example_param = next(model.parameters())
    channels = example_param.size(1) if example_param.dim() >= 2 else 3
    image_shape = (channels, 32, 32)
    return DiffusionPrior(
        image_shape=image_shape,
        steps=attack_cfg.diffusion_steps,
        device=device,
    )


def run_attack(
    config: ExperimentConfig, training_log: TrainingLog, unlearning_log: UnlearningLog
) -> AttackResult:
    fed_cfg = config.resolve_federated()
    num_classes = 10 if config.dataset.lower() != "cifar100" else 100
    model = build_model(config.dataset, num_classes)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if unlearning_log.global_models_after:
        model.load_state_dict(unlearning_log.global_models_after[-1])

    prior = _build_prior(model, device, config.attack)
    granularity = config.granularity.lower()
    images = torch.empty(0)

    if granularity == "client":
        target = build_client_level_target_signal(
            snapshots_before=training_log.global_models,
            snapshots_after=unlearning_log.global_models_after,
            target_client=config.target_id,
            client_update_log=training_log.client_updates,
        )
        images = run_client_level_attack(
            model,
            prior,
            target,
            batch_size=config.attack.attack_batch_size,
            steps=config.attack.diffusion_steps,
        )
    elif granularity == "class":
        target = build_class_level_target_signal(
            snapshots_before=training_log.global_models,
            snapshots_after=unlearning_log.global_models_after,
            target_class=config.target_id,
        )
        images = run_class_level_attack(
            model,
            prior,
            target,
            batch_size=config.attack.attack_batch_size,
            steps=config.attack.diffusion_steps,
        )
    elif granularity == "sample":
        target = build_sample_level_target_signal(
            snapshots_before=training_log.global_models,
            snapshots_after=unlearning_log.global_models_after,
        )
        images = run_sample_level_attack(
            model,
            prior,
            target,
            batch_size=config.attack.attack_batch_size,
            steps=config.attack.diffusion_steps,
            total_candidates=config.attack.total_candidates,
        )
    else:
        raise ValueError(f"Unsupported granularity: {config.granularity}")

    return AttackResult(images=images.cpu(), metadata={"attack_granularity": granularity})


def run_experiment(config: ExperimentConfig) -> ExperimentRun:
    training_log = federated_train(config)
    unlearning_log = apply_unlearning(config)
    attack_result = run_attack(config, training_log, unlearning_log)
    return ExperimentRun(
        config=config,
        training_log=training_log,
        unlearning_log=unlearning_log,
        attack_result=attack_result,
    )


def run_grid() -> List[ExperimentRun]:
    """Run the full dataset × granularity × algorithm grid."""

    results: List[ExperimentRun] = []
    for dataset in DATASETS:
        for granularity in GRANULARITIES:
            for algorithm in ALGORITHMS[granularity]:
                cfg = ExperimentConfig(
                    dataset=dataset,
                    granularity=granularity,
                    algorithm=algorithm,
                    target_id=0,
                )
                results.append(run_experiment(cfg))
    return results


__all__ = [
    "ExperimentConfig",
    "TrainingLog",
    "UnlearningLog",
    "AttackResult",
    "ExperimentRun",
    "run_experiment",
    "run_grid",
]
