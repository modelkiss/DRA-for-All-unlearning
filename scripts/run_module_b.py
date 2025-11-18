import argparse
from pathlib import Path

from federated.config import FederatedConfig, LocalDPConfig, PartitionStrategy, TrainingStopMode
from federated.pipeline import module_b_federated_unlearning


# Reuse the federated configuration so Module B knows how many clients/classes exist.
def parse_args():
    parser = argparse.ArgumentParser(description="Run Module B federated unlearning")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument("--granularity", choices=["client", "class", "sample"], default=None)
    parser.add_argument(
        "--algorithm",
        choices=["FedEraser", "FedRecovery", "FUCRT", "FUCP", "FedAF", "FedAU"],
        default=None,
    )
    parser.add_argument("--aggregation-rounds", type=int, default=5)
    parser.add_argument("--target-id", type=int, default=None, help="Target client/class/sample id")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory holding Module A checkpoints")
    parser.add_argument(
        "--save-dir", type=Path, default=Path("artifacts/module_b_models"), help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-clients", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = FederatedConfig(
        dataset=args.dataset,
        num_clients=args.num_clients,
        local_epochs=1,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        partition_strategy=PartitionStrategy.DIRICHLET_NO_REPLACEMENT,
        alpha=0.5,
        training_stop_mode=TrainingStopMode.FIXED_ROUNDS,
        max_rounds=1,
        target_accuracy=0.0,
        dp=LocalDPConfig(enabled=False),
        device=args.device,
    )

    result = module_b_federated_unlearning(
        cfg,
        granularity=args.granularity,
        algorithm=args.algorithm,
        aggregation_rounds=args.aggregation_rounds,
        target_id=args.target_id,
        model_dir=args.model_dir,
        save_dir=args.save_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
