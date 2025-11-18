import argparse

from federated.config import (
    FederatedConfig,
    LocalDPConfig,
    PartitionStrategy,
    TrainingStopMode,
)
from federated.training import run_module_a


def parse_args():
    parser = argparse.ArgumentParser(description="Run Module A federated training")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--partition-strategy",
        type=str,
        default=PartitionStrategy.DIRICHLET_NO_REPLACEMENT.value,
        choices=[s.value for s in PartitionStrategy],
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--training-stop-mode",
        type=str,
        default=TrainingStopMode.FIXED_ROUNDS.value,
        choices=[m.value for m in TrainingStopMode],
    )
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--target-accuracy", type=float, default=0.8)
    parser.add_argument("--dp", action="store_true", help="Enable local DP")
    parser.add_argument("--dp-clip", type=float, default=1.0)
    parser.add_argument("--dp-noise", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    dp_cfg = LocalDPConfig(
        enabled=args.dp, clip_norm=args.dp_clip, noise_multiplier=args.dp_noise
    )
    cfg = FederatedConfig(
        dataset=args.dataset,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        partition_strategy=PartitionStrategy(args.partition_strategy),
        alpha=args.alpha,
        training_stop_mode=TrainingStopMode(args.training_stop_mode),
        max_rounds=args.max_rounds,
        target_accuracy=args.target_accuracy,
        dp=dp_cfg,
        device=args.device,
    )

    results = run_module_a(cfg)
    print(results)


if __name__ == "__main__":
    main()
