import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import ModuleDSplit, PartitionStrategy


DATASET_NORMALIZATION = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    "cifar100": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "svhn": ([0.4377, 0.4438, 0.4728], [0.198, 0.201, 0.197]),
    "femnist": ([0.1307], [0.3081]),
    "flair": ([0.5], [0.5]),
}


@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset
    num_classes: int


class UnsupportedDatasetError(ValueError):
    pass


class UnsupportedPartitionError(ValueError):
    pass


def _default_transforms(dataset: str):
    mean, std = DATASET_NORMALIZATION[dataset]
    if dataset in {"femnist", "flair"}:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_dataset(dataset: str, root: str = "./data") -> DatasetBundle:
    name = dataset.lower()
    if name == "cifar10":
        transform = _default_transforms(name)
        train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        return DatasetBundle(train, test, num_classes=10)
    if name == "cifar100":
        transform = _default_transforms(name)
        train = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
        return DatasetBundle(train, test, num_classes=100)
    if name == "svhn":
        transform = _default_transforms(name)
        train = datasets.SVHN(root=root, split="train", download=True, transform=transform)
        test = datasets.SVHN(root=root, split="test", download=True, transform=transform)
        return DatasetBundle(train, test, num_classes=10)
    if name == "femnist":
        transform = _default_transforms(name)
        # FEMNIST is approximated by EMNIST Balanced split for easy access.
        train = datasets.EMNIST(
            root=root, split="balanced", train=True, download=True, transform=transform
        )
        test = datasets.EMNIST(
            root=root, split="balanced", train=False, download=True, transform=transform
        )
        return DatasetBundle(train, test, num_classes=47)
    if name == "flair":
        try:
            import datasets as hf_datasets  # type: ignore
        except ModuleNotFoundError as exc:
            raise UnsupportedDatasetError(
                "The FLAIR dataset loader requires the `datasets` package."
            ) from exc
        # Placeholder: use the AG News subset to approximate a FLAIR text corpus.
        ag_news = hf_datasets.load_dataset("ag_news")

        class NewsTensorDataset(Dataset):
            def __init__(self, split, max_len: int = 256):
                self.split = split
                self.max_len = max_len

            def __len__(self):
                return len(ag_news[self.split])

            def __getitem__(self, idx):
                record = ag_news[self.split][idx]
                encoded = list(record["text"].encode("utf-8"))[: self.max_len]
                padded = encoded + [0] * (self.max_len - len(encoded))
                text_tensor = torch.tensor(padded, dtype=torch.long)
                label = torch.tensor(record["label"], dtype=torch.long)
                return text_tensor, label

        return DatasetBundle(
            NewsTensorDataset("train"), NewsTensorDataset("test"), num_classes=4
        )
    raise UnsupportedDatasetError(f"Unsupported dataset: {dataset}")


def _split_for_diffusion(labels: Iterable[int], ratio: float = 0.1) -> ModuleDSplit:
    per_class_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        per_class_indices[int(label)].append(idx)

    diffusion_indices: List[int] = []
    federated_indices: List[int] = []
    for label, indices in per_class_indices.items():
        num_for_diffusion = max(1, int(len(indices) * ratio))
        random.shuffle(indices)
        diffusion_indices.extend(indices[:num_for_diffusion])
        federated_indices.extend(indices[num_for_diffusion:])
    return ModuleDSplit(diffusion_indices, federated_indices)


def _dirichlet_no_replacement(
    labels: List[int], num_clients: int, alpha: float
) -> List[List[int]]:
    labels = list(labels)
    per_class_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        per_class_indices[int(label)].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for class_indices in per_class_indices.values():
        random.shuffle(class_indices)
        proportions = torch.distributions.Dirichlet(alpha * torch.ones(num_clients)).sample()
        counts = (proportions * len(class_indices)).long()
        while counts.sum() < len(class_indices):
            counts[torch.argmin(counts)] += 1
        offset = 0
        for client_id, count in enumerate(counts.tolist()):
            client_indices[client_id].extend(class_indices[offset : offset + count])
            offset += count
    return client_indices


def _dirichlet_with_replacement(
    labels: List[int], num_clients: int, alpha: float
) -> List[List[int]]:
    labels = list(labels)
    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[int(label)].append(idx)

    classes = list(class_to_indices.keys())
    class_prior = torch.ones(len(classes))
    client_indices = [[] for _ in range(num_clients)]
    target_per_client = math.ceil(len(labels) / num_clients)
    for client_id in range(num_clients):
        class_proportions = torch.distributions.Dirichlet(alpha * class_prior).sample()
        for _ in range(target_per_client):
            sampled_class = int(torch.multinomial(class_proportions, 1).item())
            class_label = classes[sampled_class]
            sampled_index = random.choice(class_to_indices[class_label])
            client_indices[client_id].append(sampled_index)
    return client_indices


def _iid_no_replacement(total_size: int, num_clients: int) -> List[List[int]]:
    indices = list(range(total_size))
    random.shuffle(indices)
    shard = int(math.ceil(total_size / num_clients))
    return [indices[i * shard : (i + 1) * shard] for i in range(num_clients)]


def partition_dataset(
    labels: Iterable[int], strategy: PartitionStrategy, num_clients: int, alpha: float
) -> List[List[int]]:
    labels_list = list(labels)
    if strategy == PartitionStrategy.DIRICHLET_NO_REPLACEMENT:
        return _dirichlet_no_replacement(labels_list, num_clients, alpha)
    if strategy == PartitionStrategy.DIRICHLET_WITH_REPLACEMENT:
        return _dirichlet_with_replacement(labels_list, num_clients, alpha)
    if strategy == PartitionStrategy.IID_NO_REPLACEMENT:
        return _iid_no_replacement(len(labels_list), num_clients)
    raise UnsupportedPartitionError(f"Unsupported strategy: {strategy}")


def build_data_loaders(
    bundle: DatasetBundle,
    cfg,
    shuffle: bool = True,
) -> Tuple[List[DataLoader], DataLoader, ModuleDSplit]:
    labels = []
    for _, label in bundle.train:
        if torch.is_tensor(label):
            labels.append(int(label.item()))
        else:
            labels.append(int(label))
    split = _split_for_diffusion(labels, ratio=0.1)
    federated_indices = split.federated_indices
    diffusion_indices = split.diffusion_indices

    client_splits = partition_dataset(
        [labels[idx] for idx in federated_indices],
        cfg.partition_strategy,
        cfg.num_clients,
        cfg.alpha,
    )

    # Map back to original dataset indices
    adjusted_splits: List[List[int]] = []
    for split_indices in client_splits:
        adjusted_splits.append([federated_indices[idx] for idx in split_indices if idx < len(federated_indices)])

    client_loaders = [
        DataLoader(Subset(bundle.train, indices), batch_size=cfg.batch_size, shuffle=shuffle)
        for indices in adjusted_splits
    ]
    test_loader = DataLoader(bundle.test, batch_size=cfg.batch_size, shuffle=False)
    return client_loaders, test_loader, ModuleDSplit(diffusion_indices, federated_indices)
