# Experiment Management & Reproducibility

TrustBench provides local experiment tracking, lineage recording, and reproducibility verification without requiring a hosted platform.

---

## Storage Backends

TrustBench offers two swappable storage implementations:

1. **Filesystem Storage (`FilesystemExperimentStore`)**:
   - Zero-database dependency.
   - Atomically organizes runs into a transparent folder structure:
     ```text
     artifacts/experiments/<experiment_id>/
     ├── manifest.json       # Experiment status, lifecycle, and SHA-256 hashes
     ├── metadata.json       # Model, dataset, environment, and hardware info
     ├── metrics.json        # Structured scalar evaluation metrics
     ├── config.json         # Exact configuration snapshot
     └── predictions.npz     # Compressed NumPy arrays (predictions, targets, probabilities)
     ```

2. **SQLite Storage (`SQLiteExperimentStore`)**:
   - Relational persistence using SQLAlchemy.
   - Enables fast multi-experiment queries, filtering, and cross-run aggregations.

---

## Lineage & Provenance Tracking

TrustBench automatically captures comprehensive lineage to guarantee reproducibility:
- **Model Checkpoint SHA-256**: Cryptographic digest of model weight files.
- **Dataset Fingerprint**: SHA-256 of dataset sample IDs and labels.
- **Random Seeds**: Global seeds recorded for Python, NumPy, and PyTorch.
- **Environment & Hardware**: Python version, OS platform, GPU driver & device name.
- **Execution Timestamps**: ISO 8601 creation and completion timestamps.

---

## Verifying Integrity

```python
from trustbench.storage.filesystem import FilesystemExperimentStore

store = FilesystemExperimentStore("artifacts/experiments")
checksums = store.verify_checksums("my_experiment_id")

for artifact, is_valid in checksums.items():
    print(f"{artifact}: {'VALID' if is_valid else 'CORRUPTED'}")
```

---

## Exporting & Importing Experiments

Experiments can be exported to standalone, portable `.tar.gz` archives:

```python
# Export
archive_path = store.export("my_experiment_id", "my_experiment.tar.gz")

# Import safely with ZipSlip protection
new_record = store.import_experiment("my_experiment.tar.gz")
```
