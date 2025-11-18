# DRA-for-All-unlearning

This repository scaffolds the DRA-for-All unlearning pipeline with separately
runnable modules and a composable pipeline entry point. Module A is fully
implemented and provides:

- Dataset handling for CIFAR10, CIFAR100, FEMNIST (EMNIST-balanced), SVHN, and
  a placeholder FLAIR text dataset.
- Hold-out of 10% per class for diffusion pretraining (Module D) while the rest
  feeds the federated loop.
- Client allocation via Dirichlet (with/without replacement) or IID splits.
- FedAvg aggregation with optional accuracy-based early stopping or fixed
  rounds, plus optional local differential privacy.
- Trajectory logging and persisting the last three global models.

Other modules (B–G) are stubbed in `src/federated/pipeline.py` so they can be
filled incrementally without blocking Module A usage.

## Running Module A

```bash
PYTHONPATH=./src \
python -m scripts.run_module_a \
  --dataset cifar10 \
  --num-clients 10 \
  --local-epochs 5 \
  --batch-size 64 \
  --lr 0.01 \
  --partition-strategy dirichlet_no_replacement \
  --max-rounds 5 \
  --target-accuracy 0.7
```

Key flags:

- `--training-stop-mode`: `fixed_rounds` or `target_accuracy`.
- `--dp`: toggles local DP (per-step gradient clipping + Gaussian noise).
- `--partition-strategy`: `dirichlet_no_replacement`,
  `dirichlet_with_replacement`, or `iid_no_replacement`.
- `--device`: defaults to `cuda` if available, otherwise CPU is used.

Artifacts:

- Aggregation trajectories: `artifacts/trajectories/round_*.pt`.
- Last three global checkpoints: `artifacts/module_a_models/`.
- Diffusion hold-out indices: returned from the runner and usable by Module D.

## Full pipeline stub

The `federated.pipeline.run_full_pipeline` function wires the module skeletons
and currently invokes Module A. As additional modules are implemented, their
outputs can be chained through this entry point without changing Module A.
