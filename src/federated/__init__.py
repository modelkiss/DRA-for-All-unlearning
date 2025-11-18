"""Federated learning toolkit for DRA-for-All unlearning pipeline.

This package organizes the code into modules aligned with the larger
unlearning pipeline. Module A is fully implemented to support data
partitioning and FedAvg training with trajectory logging; downstream modules
are scaffolded in :mod:`federated.pipeline`.
"""

__all__ = [
    "config",
    "data",
    "models",
    "training",
    "pipeline",
]
