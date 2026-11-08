"""Example 7: Save, Retrieve, and Reproduce an Evaluation Experiment.

Demonstrates tracking model lineage (hashes, seeds, environment), atomically
persisting experiment artifacts, and re-loading results using FilesystemExperimentStore.
"""

import tempfile
from pathlib import Path

import numpy as np

from trustbench.core.results import EvaluationResult
from trustbench.storage.filesystem import FilesystemExperimentStore


def main():
    print("--- Example 7: Save and Reproduce Experiment ---")

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "experiments"
        store = FilesystemExperimentStore(base_dir=store_path)

        # 1. Prepare evaluation output
        result = EvaluationResult(
            metrics={"classification": {"accuracy": 0.954, "macro_f1": 0.951}},
            predictions=np.array([1, 0, 1, 1]),
            labels=np.array([1, 0, 1, 0]),
            sample_ids=np.array(["s1", "s2", "s3", "s4"]),
            probabilities=np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.4, 0.6]]),
        )

        exp_id = "exp_cifar10_demo"
        # 2. Persist experiment & artifacts
        print(f"Creating experiment {exp_id} in filesystem store...")
        store.create(
            name="CIFAR10 ResNet18 Run",
            experiment_id=exp_id,
            config={"seed": 42, "batch_size": 128},
            metadata={"name": "ResNet18", "checkpoint_sha256": "abc123def456"},
        )
        store.save_result(exp_id, result)

        # 3. Retrieve experiment
        retrieved = store.get(exp_id)
        assert retrieved is not None
        print(f"Successfully retrieved: {retrieved.name} (Status: {retrieved.status})")
        print(f"Recorded Accuracy: {retrieved.metrics['classification']['accuracy']}")

        # 4. Verify checksum integrity of all artifacts
        checksum_results = store.verify_checksums(exp_id)
        for artifact, is_valid in checksum_results.items():
            print(f"Artifact {artifact}: Checksum Valid = {is_valid}")
            assert is_valid, f"Artifact {artifact} checksum failed!"

        # 5. Load persisted artifacts
        from trustbench.reporting.artifacts import load_experiment_artifacts

        exp_dir = store_path / exp_id
        loaded_artifacts = load_experiment_artifacts(exp_dir)
        print(f"Loaded {len(loaded_artifacts['predictions']['predictions'])} predictions from disk.")
        print("✓ Experiment save, checksum verification, and reproducibility verified.")


if __name__ == "__main__":
    main()
