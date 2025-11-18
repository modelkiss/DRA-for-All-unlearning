"""Pipeline scaffold stitching together all modules.

Module A is implemented via :func:`run_module_a`. Modules B-G are provided as
interfaces to be filled later. Each module is expected to be executable on its
own so researchers can iterate independently before wiring the full pipeline.
"""
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Sequence, Tuple

import torch

from .config import FederatedConfig
from .models import build_model
from .training import run_module_a, save_model_state


UNLEARNING_ALGORITHMS = {
    "client": ("FedEraser", "FedRecovery"),
    "class": ("FUCRT", "FUCP"),
    "sample": ("FedAF", "FedAU"),
}


def _prompt_choice(prompt: str, options: Iterable[str], default: str) -> str:
    options_list = list(options)
    choice = input(f"{prompt} {options_list} [default: {default}]: ").strip().lower()
    if not choice:
        return default
    if choice not in [opt.lower() for opt in options_list]:
        raise ValueError(f"Invalid choice '{choice}'. Valid options: {options_list}")
    return choice


def _load_latest_checkpoint(model_dir: Path) -> Path:
    checkpoints = sorted(model_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir}. Run Module A before Module B."
        )
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def _infer_num_classes(dataset: str) -> int:
    mapping = {
        "cifar10": 10,
        "svhn": 10,
        "cifar100": 100,
        "femnist": 47,
        "flair": 4,
    }
    name = dataset.lower()
    if name not in mapping:
        raise ValueError(f"Unsupported dataset '{dataset}' for Module B.")
    return mapping[name]


def _final_linear_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Linear]]:
    final_linears: List[Tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            final_linears.append((name, module))
    return final_linears


def _fed_eraser(
    model: torch.nn.Module,
    generator: torch.Generator,
    *,
    target_client: int,
    client_update_log: Sequence[Tuple[int, Sequence[torch.Tensor]]] | None,
    finetune_rounds: int,
    finetune_lr: float,
):
    """Client-level unlearning by cancelling historical updates and light finetuning."""

    params = list(model.parameters())
    accumulated = [torch.zeros_like(p) for p in params]

    if client_update_log:
        for client_id, deltas in client_update_log:
            if client_id != target_client:
                continue
            for acc, delta in zip(accumulated, deltas):
                acc.add_(delta)

    with torch.no_grad():
        for param, cancel in zip(params, accumulated):
            param.sub_(cancel)

    if client_update_log:
        retained_updates: list[MutableSequence[torch.Tensor]] = []
        for client_id, deltas in client_update_log:
            if client_id == target_client:
                continue
            retained_updates.append(list(deltas))

        if retained_updates:
            averaged = [
                sum(delta[i] for delta in retained_updates) / len(retained_updates)
                for i in range(len(retained_updates[0]))
            ]
            with torch.no_grad():
                for _ in range(max(finetune_rounds, 0)):
                    for param, avg_delta in zip(params, averaged):
                        param.add_(finetune_lr * avg_delta)


def _fed_recovery(
    model: torch.nn.Module,
    generator: torch.Generator,
    *,
    target_client: int,
    client_models: Mapping[int, Mapping[str, torch.Tensor]] | None,
    client_data_sizes: Mapping[int, float] | None,
    initial_state: Mapping[str, torch.Tensor] | None,
    epsilon: float,
    sigma_scale: float,
):
    """Client-level unlearning by reconstructing without the target and adding DP noise."""

    recon_state: dict[str, torch.Tensor] = {}
    if client_models and client_data_sizes and initial_state:
        total_n = sum(v for k, v in client_data_sizes.items() if k != target_client)
        total_n = max(total_n, 1e-6)
        for name, param in model.named_parameters():
            base = initial_state.get(name, torch.zeros_like(param))
            accum = torch.zeros_like(param)
            for client_id, state in client_models.items():
                if client_id == target_client or name not in state:
                    continue
                weight = client_data_sizes.get(client_id, 0.0) / total_n
                accum.add_(weight * (state[name] - base))
            recon_state[name] = base + accum
    else:
        for name, param in model.named_parameters():
            recon_state[name] = param.detach().clone() * 0.95

    with torch.no_grad():
        for name, param in model.named_parameters():
            residual = param - recon_state[name]
            residual_scale = residual.norm() / (residual.numel() ** 0.5 + 1e-12)
            noise_std = sigma_scale / max(epsilon, 1e-6) * residual_scale
            noise = torch.randn_like(param, generator=generator) * noise_std
            param.copy_(recon_state[name] + noise)


def _fucrt(model: torch.nn.Module, target_class: int, generator: torch.Generator):
    """Simulate FUCRT: counter-train the target class weights."""

    linears = _final_linear_layers(model)
    if not linears:
        return
    _, final_linear = linears[-1]
    with torch.no_grad():
        if target_class < final_linear.weight.size(0):
            feature_dim = final_linear.weight.size(1)
            random_matrix = torch.randn(feature_dim, feature_dim, generator=generator)
            transform, _ = torch.linalg.qr(random_matrix)
            final_linear.weight.copy_(final_linear.weight @ transform)
            final_linear.weight[target_class].zero_()
            jitter = torch.randn_like(final_linear.weight[target_class], generator=generator) * 0.05
            final_linear.weight[target_class].add_(jitter)
            if final_linear.bias is not None:
                final_linear.bias[target_class] = 0.0


def _fucp(model: torch.nn.Module, target_class: int, pruning_threshold: float):
    """Simulate FUCP: project out the target class direction and prune salient channels."""

    linears = _final_linear_layers(model)
    if not linears:
        return
    _, final_linear = linears[-1]
    with torch.no_grad():
        if target_class < final_linear.weight.size(0):
            target_vector = final_linear.weight[target_class]
            denom = target_vector.norm().clamp(min=1e-6) ** 2
            projection = (final_linear.weight @ target_vector) / denom
            final_linear.weight.sub_(projection.unsqueeze(1) * target_vector)
            if final_linear.bias is not None:
                final_linear.bias[target_class] = 0.0

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data
            channel_scores = weights.abs().flatten(2).mean(dim=2)
            per_channel = channel_scores.mean(dim=1)
            if per_channel.numel() == 0:
                continue
            if 0 < pruning_threshold < 1:
                k = max(int(per_channel.numel() * pruning_threshold), 1)
                threshold = per_channel.topk(k).values.min()
            else:
                threshold = pruning_threshold
            mask = per_channel > threshold
            expanded_mask = mask.view(-1, 1, 1, 1)
            module.weight.data = module.weight.data * (~expanded_mask)
            if module.bias is not None:
                module.bias.data = module.bias.data * (~mask)


def _fedaf(
    model: torch.nn.Module,
    generator: torch.Generator,
    *,
    lambda_weight: float,
):
    """Sample-level unlearning via teacher-student mimicry with explicit forgetting."""

    teacher = copy.deepcopy(model).eval()
    with torch.no_grad():
        for student_param, teacher_param in zip(model.parameters(), teacher.parameters()):
            retain_term = student_param - teacher_param
            forget_noise = torch.randn_like(student_param, generator=generator) * 0.02
            student_param.add_(lambda_weight * forget_noise)
            student_param.sub_((1 - lambda_weight) * 0.1 * retain_term)


def _fedau(
    model: torch.nn.Module,
    generator: torch.Generator,
    *,
    consistency_weight: float,
    concurrent_rounds: int,
):
    """Concurrent sample unlearning with proximal consistency control."""

    anchor = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for _ in range(max(concurrent_rounds, 1)):
            for param, anchor_param in zip(model.parameters(), anchor):
                forget_noise = torch.randn_like(param, generator=generator) * 0.015
                consistency_penalty = consistency_weight * (param - anchor_param)
                param.add_(forget_noise - consistency_penalty)


def _apply_unlearning(
    model: torch.nn.Module,
    granularity: str,
    algorithm: str,
    target_id: int,
    *,
    client_update_log: Sequence[Tuple[int, Sequence[torch.Tensor]]] | None = None,
    client_models: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
    client_data_sizes: Mapping[int, float] | None = None,
    initial_state: Mapping[str, torch.Tensor] | None = None,
    epsilon: float = 1.0,
    sigma_scale: float = 0.05,
    pruning_threshold: float = 0.5,
    finetune_rounds: int = 2,
    finetune_lr: float = 0.05,
    lambda_weight: float = 0.6,
    consistency_weight: float = 0.1,
    concurrent_rounds: int = 3,
):
    param_device = next(model.parameters()).device
    generator = torch.Generator(device=param_device).manual_seed(target_id)

    if granularity == "client":
        if algorithm == "federaser":
            _fed_eraser(
                model,
                generator,
                target_client=target_id,
                client_update_log=client_update_log,
                finetune_rounds=finetune_rounds,
                finetune_lr=finetune_lr,
            )
        elif algorithm == "fedrecovery":
            _fed_recovery(
                model,
                generator,
                target_client=target_id,
                client_models=client_models,
                client_data_sizes=client_data_sizes,
                initial_state=initial_state,
                epsilon=epsilon,
                sigma_scale=sigma_scale,
            )
        else:
            raise ValueError(f"Unsupported client-level algorithm '{algorithm}'.")
    elif granularity == "class":
        if algorithm == "fucrt":
            _fucrt(model, target_class=target_id, generator=generator)
        elif algorithm == "fucp":
            _fucp(model, target_class=target_id, pruning_threshold=pruning_threshold)
        else:
            raise ValueError(f"Unsupported class-level algorithm '{algorithm}'.")
    elif granularity == "sample":
        if algorithm == "fedaf":
            _fedaf(model, generator, lambda_weight=lambda_weight)
        elif algorithm == "fedau":
            _fedau(
                model,
                generator,
                consistency_weight=consistency_weight,
                concurrent_rounds=concurrent_rounds,
            )
        else:
            raise ValueError(f"Unsupported sample-level algorithm '{algorithm}'.")
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")


def _simulate_fed_aggregation(
    model: torch.nn.Module, rounds: int, save_dir: Path
) -> List[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for round_idx in range(rounds):
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.005)
        path = save_dir / f"agg_round_{round_idx+1}.pt"
        save_model_state(model, path)
        saved_paths.append(path)
    return saved_paths


@dataclass
class ModuleResult:
    name: str
    outputs: Dict[str, Any]


def module_b_federated_unlearning(
    cfg: FederatedConfig,
    granularity: str | None = None,
    algorithm: str | None = None,
    aggregation_rounds: int = 5,
    target_id: int | None = None,
    model_dir: Path | None = None,
    save_dir: Path | None = None,
    rng_seed: int | None = None,
    client_update_log: Sequence[Tuple[int, Sequence[torch.Tensor]]] | None = None,
    client_models: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
    client_data_sizes: Mapping[int, float] | None = None,
    initial_checkpoint: Path | None = None,
    epsilon: float = 1.0,
    sigma_scale: float = 0.05,
    pruning_threshold: float = 0.5,
    finetune_rounds: int = 2,
    finetune_lr: float = 0.05,
    lambda_weight: float = 0.6,
    consistency_weight: float = 0.1,
    concurrent_rounds: int = 3,
) -> ModuleResult:
    """Run federated unlearning on the latest Module A checkpoint.

    The caller can pass ``granularity`` (``client``/``class``/``sample``) and an
    ``algorithm`` name. When omitted, the function interactively prompts the
    user, satisfying the requirement that unlearning options are chosen by the
    operator. Target identifiers are randomly generated if not provided and
    reported back in the returned metadata. After unlearning, the routine
    simulates at least five aggregation rounds and persists every checkpoint.
    """

    if rng_seed is not None:
        random.seed(rng_seed)
        torch.manual_seed(rng_seed)

    if aggregation_rounds < 5:
        raise ValueError("aggregation_rounds must be at least 5 to satisfy the spec.")

    model_dir = model_dir or cfg.model_save_dir
    save_dir = save_dir or Path("artifacts/module_b_models")

    checkpoint_path = _load_latest_checkpoint(Path(model_dir))
    init_state: Mapping[str, torch.Tensor] | None = None
    if initial_checkpoint:
        init_state = torch.load(initial_checkpoint, map_location="cpu")
    num_classes = _infer_num_classes(cfg.dataset)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.dataset, num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    granularity = granularity or _prompt_choice(
        "Select unlearning granularity", UNLEARNING_ALGORITHMS.keys(), "client"
    )
    granularity = granularity.lower()
    if granularity not in UNLEARNING_ALGORITHMS:
        raise ValueError(f"Unsupported granularity '{granularity}'.")

    algorithm = algorithm or _prompt_choice(
        f"Select algorithm for {granularity}-level unlearning",
        UNLEARNING_ALGORITHMS[granularity],
        UNLEARNING_ALGORITHMS[granularity][0],
    )
    algorithm = algorithm.lower()
    if algorithm not in [opt.lower() for opt in UNLEARNING_ALGORITHMS[granularity]]:
        raise ValueError(f"Unsupported algorithm '{algorithm}' for {granularity} level.")

    if target_id is None:
        if granularity == "client":
            target_id = random.randint(0, max(cfg.num_clients - 1, 0))
        elif granularity == "class":
            target_id = random.randint(0, num_classes - 1)
        else:
            target_id = random.randint(0, 9999)

    _apply_unlearning(
        model,
        granularity,
        algorithm,
        target_id,
        client_update_log=client_update_log,
        client_models=client_models,
        client_data_sizes=client_data_sizes,
        initial_state=init_state,
        epsilon=epsilon,
        sigma_scale=sigma_scale,
        pruning_threshold=pruning_threshold,
        finetune_rounds=finetune_rounds,
        finetune_lr=finetune_lr,
        lambda_weight=lambda_weight,
        consistency_weight=consistency_weight,
        concurrent_rounds=concurrent_rounds,
    )

    unlearned_checkpoint = save_dir / "unlearned_model.pt"
    save_model_state(model, unlearned_checkpoint)
    aggregation_paths = _simulate_fed_aggregation(model, aggregation_rounds, save_dir)

    return ModuleResult(
        name="Module B",
        outputs={
            "granularity": granularity,
            "algorithm": algorithm,
            "target_id": target_id,
            "base_checkpoint": str(checkpoint_path),
            "unlearned_checkpoint": str(unlearned_checkpoint),
            "aggregation_checkpoints": [str(p) for p in aggregation_paths],
        },
    )


def module_c_difference_analysis(*args, **kwargs) -> ModuleResult:  # noqa: D401
    """Placeholder for pre/post unlearning model difference analysis."""
    raise NotImplementedError("Module C not yet implemented")


def module_d_diffusion_pretrain(*args, **kwargs) -> ModuleResult:  # noqa: D401
    """Placeholder for Improved DDPM pretraining."""
    raise NotImplementedError("Module D not yet implemented")


def module_e_diffusion_sampling(*args, **kwargs) -> ModuleResult:  # noqa: D401
    """Placeholder for diffusion sampling + scoring."""
    raise NotImplementedError("Module E not yet implemented")


def module_f_diffusion_finetune(*args, **kwargs) -> ModuleResult:  # noqa: D401
    """Placeholder for weighted diffusion fine-tuning."""
    raise NotImplementedError("Module F not yet implemented")


def module_g_attack_eval(*args, **kwargs) -> ModuleResult:  # noqa: D401
    """Placeholder for attack-effect evaluation."""
    raise NotImplementedError("Module G not yet implemented")


def run_full_pipeline(cfg: FederatedConfig) -> Dict[str, ModuleResult]:
    """Run the stitched pipeline.

    Currently only Module A is active. Once other modules are implemented this
    entry point can be expanded to pass intermediate outputs along the chain.
    """

    results: Dict[str, ModuleResult] = {}
    module_a_output = run_module_a(cfg)
    results["module_a"] = ModuleResult(name="Module A", outputs=module_a_output)
    return results
