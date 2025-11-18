import copy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .config import FederatedConfig, TrainingStopMode
from .data import build_data_loaders, load_dataset
from .models import build_model


def _apply_local_dp(model: nn.Module, dp_cfg):
    if not dp_cfg.enabled:
        return
    clip_grad_norm_(model.parameters(), dp_cfg.clip_norm)
    for param in model.parameters():
        if param.grad is None:
            continue
        noise = torch.randn_like(param.grad) * dp_cfg.noise_multiplier * dp_cfg.clip_norm
        param.grad += noise


def _train_one_client(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    dp_cfg,
    local_epochs: int,
) -> int:
    model.train()
    num_samples = 0
    for _ in range(local_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            _apply_local_dp(model, dp_cfg)
            optimizer.step()
            num_samples += targets.size(0)
    return num_samples


def _evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total if total else 0.0


def _fed_avg(client_states: List[Tuple[Dict[str, torch.Tensor], int]]):
    total_samples = sum(num for _, num in client_states)
    if total_samples == 0:
        raise ValueError("No samples available for FedAvg aggregation.")
    aggregated_state: Dict[str, torch.Tensor] = {}
    for state_dict, num_samples in client_states:
        weight = num_samples / total_samples
        for key, tensor in state_dict.items():
            if key not in aggregated_state:
                aggregated_state[key] = tensor.detach().clone() * weight
            else:
                aggregated_state[key] += tensor.detach().clone() * weight
    return aggregated_state


def save_model_state(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def run_module_a(cfg: FederatedConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset_bundle = load_dataset(cfg.dataset)
    client_loaders, test_loader, split = build_data_loaders(dataset_bundle, cfg)
    model = build_model(cfg.dataset, dataset_bundle.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    trajectory_dir = Path(cfg.trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    last_rounds: List[Path] = []

    for round_idx in range(cfg.max_rounds):
        client_states: List[Tuple[Dict[str, torch.Tensor], int]] = []
        for client_id, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(model)
            optimizer = optim.SGD(
                local_model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum
            )
            num_samples = _train_one_client(
                local_model,
                loader,
                criterion,
                optimizer,
                device,
                cfg.dp,
                cfg.local_epochs,
            )
            client_states.append((local_model.state_dict(), num_samples))

        aggregated_state = _fed_avg(client_states)
        model.load_state_dict(aggregated_state)
        accuracy = _evaluate(model, test_loader, device)
        trajectory_path = trajectory_dir / f"round_{round_idx+1}.pt"
        save_model_state(model, trajectory_path)

        if len(last_rounds) >= 3:
            last_rounds.pop(0)
        last_rounds.append(trajectory_path)

        if cfg.training_stop_mode == TrainingStopMode.TARGET_ACCURACY and accuracy >= cfg.target_accuracy:
            print(
                f"Reached target accuracy {accuracy:.4f} at round {round_idx+1}; stopping early."
            )
            break
        print(f"Round {round_idx+1}: accuracy={accuracy:.4f}")

    cfg.model_save_dir.mkdir(parents=True, exist_ok=True)
    for path in last_rounds:
        destination = cfg.model_save_dir / path.name
        destination.write_bytes(path.read_bytes())
    return {
        "final_accuracy": _evaluate(model, test_loader, device),
        "diffusion_indices": split.diffusion_indices,
        "federated_indices": split.federated_indices,
        "saved_models": [str(p) for p in last_rounds],
    }
