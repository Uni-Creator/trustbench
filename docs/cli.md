# Command Line Interface (CLI) Reference

TrustBench provides a Click-powered CLI executable via `trustbench`.

---

## Commands Overview

| Command | Usage | Description |
|---|---|---|
| `version` | `trustbench version` | Print installed package version |
| `init` | `trustbench init [-o PATH] [--force]` | Scaffold a `trustbench.yaml` configuration |
| `validate` | `trustbench validate [CONFIG]` | Check syntax and schema of configuration file |
| `evaluate` | `trustbench evaluate [CONFIG] [OPTIONS]` | Execute evaluation pipeline from config |
| `report` | `trustbench report --artifacts DIR` | Generate standalone HTML and JSON reports |
| `compare` | `trustbench compare EXP_A EXP_B` | Compare two experiment directories |

---

## 1. `trustbench version`
Prints the installed package version.

```bash
$ trustbench version
trustbench 0.2.0
```

---

## 2. `trustbench init`
Generates a starter configuration file with recommended evaluation defaults.

```bash
# Creates trustbench.yaml in current directory
trustbench init

# Specify custom file name
trustbench init --output custom_eval.yaml

# Overwrite existing file
trustbench init --force
```

---

## 3. `trustbench validate`
Validates model paths, dataset settings, metric groups, and reporting parameters without running inference.

```bash
$ trustbench validate trustbench.yaml
✓ trustbench.yaml is valid.
```

If validation fails, specific syntax or path errors are displayed with actionable hints.

---

## 4. `trustbench evaluate`
Executes model inference and metric computation as specified in the configuration file.

```bash
# Evaluate using default trustbench.yaml
trustbench evaluate

# Pass config path as positional argument
trustbench evaluate configs/cifar10_resnet18.yaml

# Pass config path via flag
trustbench evaluate --config configs/cifar10_resnet18.yaml

# Override output directory
trustbench evaluate trustbench.yaml --output-dir ./results/run_1

# Quiet or verbose modes
trustbench evaluate trustbench.yaml --quiet
trustbench evaluate trustbench.yaml --verbose
```
