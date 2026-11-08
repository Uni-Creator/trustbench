# Configuration Guide (`trustbench.yaml`)

TrustBench uses clean YAML configuration files to define reproducible evaluation experiments.

---

## Configuration Specification

```yaml
config_version: 1
experiment_name: resnet18_cifar10_evaluation
seed: 42

model:
  checkpoint_path: checkpoint/ckpt.pth  # Path to model weights
  arch_name: ResNet18                   # Architecture name
  num_classes: 10                       # Number of classes
  device: cuda:0                        # null for auto-detect, or 'cpu', 'cuda', 'cuda:0'

dataset:
  data_dir: dataset                     # Root directory of dataset
  dataset_type: cifar10                 # Dataset loader type ('cifar10', etc.)
  batch_size: 128                       # Evaluation batch size
  num_workers: 2                        # DataLoader worker processes
  split: test                           # Dataset split ('train', 'test', 'val')

metrics:
  groups:
    - classification                    # Accuracy, Macro-F1, Precision, Recall
    - calibration                       # ECE, Adaptive ECE, Brier score, NLL
    - uncertainty                       # Predictive entropy, Free energy, MSP
    - error_detection                   # AUROC, AUPRC, FPR@95%TPR
    - selective_prediction              # Risk-Coverage curves, AURC
  temperature: null                     # Optional temperature scaling factor

robustness:
  enabled: false                        # Set to true to run corruption suites
  corruption_types:
    - gaussian_noise
    - brightness
    - gaussian_blur
  severity_levels: [1, 2, 3, 4, 5]

reporting:
  output_dir: artifacts/experiments/resnet18_cifar10
  save_html: true                       # Generate standalone interactive HTML report
  save_json: true                       # Generate versioned machine-readable JSON
```

---

## Schema Fields & Types

### Root Options
- `config_version` *(int, required)*: Version of the configuration schema (current: `1`).
- `experiment_name` *(str, required)*: Unique identifier or human-readable title.
- `seed` *(int, optional)*: Random seed applied to Python, NumPy, and PyTorch.

### Model Section (`model`)
- `checkpoint_path` *(str, required)*: Relative or absolute path to weights.
- `arch_name` *(str, required)*: Registered model architecture name.
- `num_classes` *(int, required)*: Positive integer count of output classes.
- `device` *(str, optional)*: Target execution device (`cpu`, `cuda`, `cuda:1`, or `null`).

### Dataset Section (`dataset`)
- `data_dir` *(str, required)*: Storage location of dataset files.
- `dataset_type` *(str, required)*: Dataset provider name (e.g., `cifar10`).
- `batch_size` *(int, default: 100)*: Batch size for batched inference.
- `num_workers` *(int, default: 0)*: Number of parallel workers for data loading.
- `split` *(str, default: 'test')*: Split identifier.

### Reporting Section (`reporting`)
- `output_dir` *(str, default: 'artifacts')*: Destination directory for generated artifacts.
- `save_html` *(bool, default: true)*: Standalone HTML report toggle.
- `save_json` *(bool, default: true)*: Standard JSON report toggle.
