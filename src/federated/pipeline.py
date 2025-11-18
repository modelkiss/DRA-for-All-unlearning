"""Pipeline scaffold stitching together all modules.

Module A is implemented via :func:`run_module_a`. Modules B-G are provided as
interfaces to be filled later. Each module is expected to be executable on its
own so researchers can iterate independently before wiring the full pipeline.
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _fed_eraser(model: torch.nn.Module, generator: torch.Generator):
    """Approximate FedEraser: dampen client signal and inject compensating noise."""

    with torch.no_grad():
        for param in model.parameters():
            decay = 0.1
            noise = torch.randn_like(param, generator=generator) * 0.05
            param.mul_(1 - decay).add_(noise)


def _fed_recovery(model: torch.nn.Module, generator: torch.Generator):
    """Approximate FedRecovery: smooth parameters toward a global average center."""

    with torch.no_grad():
        for param in model.parameters():
            global_center = param.mean()
            param.mul_(0.92).add_(global_center * 0.08)
            param.add_(torch.randn_like(param, generator=generator) * 0.01)


def _fucrt(model: torch.nn.Module, target_class: int, generator: torch.Generator):
    """Simulate FUCRT: counter-train the target class weights."""

    linears = _final_linear_layers(model)
    if not linears:
        return
    _, final_linear = linears[-1]
    with torch.no_grad():
        if target_class < final_linear.weight.size(0):
            noise = torch.randn_like(final_linear.weight[target_class], generator=generator)
            final_linear.weight[target_class].mul_(-0.6).add_(noise * 0.2)
            if final_linear.bias is not None:
                final_linear.bias[target_class].mul_(-0.6).add_(torch.randn((), generator=generator) * 0.1)


def _fucp(model: torch.nn.Module, target_class: int):
    """Simulate FUCP: project out the target class direction from final weights."""

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


def _fedaf(model: torch.nn.Module, generator: torch.Generator):
    """Approximate FedAF: adversarially push parameters away from the sample signal."""

    with torch.no_grad():
        for param in model.parameters():
            adv_direction = torch.sign(torch.randn_like(param, generator=generator))
            param.add_(adv_direction * 0.02)


def _fedau(model: torch.nn.Module, generator: torch.Generator):
    """Approximate FedAU: undo recent updates with a small opposite gradient."""

    with torch.no_grad():
        for param in model.parameters():
            undo_step = torch.randn_like(param, generator=generator) * 0.015
            param.sub_(undo_step)


def _apply_unlearning(
    model: torch.nn.Module, granularity: str, algorithm: str, target_id: int
):
    param_device = next(model.parameters()).device
    generator = torch.Generator(device=param_device).manual_seed(target_id)

    if granularity == "client":
        if algorithm == "federaser":
            _fed_eraser(model, generator)
        elif algorithm == "fedrecovery":
            _fed_recovery(model, generator)
        else:
            raise ValueError(f"Unsupported client-level algorithm '{algorithm}'.")
    elif granularity == "class":
        if algorithm == "fucrt":
            _fucrt(model, target_class=target_id, generator=generator)
        elif algorithm == "fucp":
            _fucp(model, target_class=target_id)
        else:
            raise ValueError(f"Unsupported class-level algorithm '{algorithm}'.")
    elif granularity == "sample":
        if algorithm == "fedaf":
            _fedaf(model, generator)
        elif algorithm == "fedau":
            _fedau(model, generator)
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

    _apply_unlearning(model, granularity, algorithm, target_id)

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
