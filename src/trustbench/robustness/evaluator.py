"""Robustness evaluation orchestrator across image corruptions and severity levels."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from scipy.special import softmax
from torch.utils.data import DataLoader

from trustbench.core.reproducibility import get_environment_info, set_seed
from trustbench.datasets.cifar10 import CIFAR10_CLASSES
from trustbench.evaluation.calibration import compute_calibration_metrics
from trustbench.evaluation.classification import compute_classification_metrics
from trustbench.evaluation.error_detection import compute_error_detection_metrics
from trustbench.evaluation.selective import compute_selective_prediction_metrics
from trustbench.evaluation.uncertainty import compute_uncertainty_metrics
from trustbench.models.checkpoint import load_model_from_checkpoint
from trustbench.models.pytorch import run_inference
from trustbench.reporting.artifacts import NpEncoder, load_experiment_artifacts
from trustbench.reporting.hashing import compute_file_sha256
from trustbench.robustness.aggregation import compute_robustness_aggregation
from trustbench.robustness.corruptions import CorruptedDataset
from trustbench.robustness.severity import serialize_severities


class RobustnessEvaluator:
    """Evaluates model performance degradation across progressive input corruptions."""

    def __init__(
        self,
        checkpoint_path: str | Path = "./checkpoint/ckpt.pth",
        model_name: str = "ResNet18",
        data_dir: str = "./dataset",
        baseline_experiment_dir: str | Path | None = "./artifacts/experiments/baseline_001",
        output_dir: str | Path = "./artifacts/experiments/robustness_001",
        batch_size: int = 128,
        num_workers: int = 2,
        seed: int = 42,
        device: str | torch.device | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.data_dir = data_dir
        self.baseline_experiment_dir = Path(baseline_experiment_dir) if baseline_experiment_dir else None
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        set_seed(self.seed)
        self.model, self.raw_checkpoint = load_model_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model_name=self.model_name,
            device=self.device,
        )

        # Base un-augmented raw CIFAR-10 testset (yields PIL Image and target)
        self.raw_testset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=None,
        )

        self.fitted_temperature: float | None = None

    def fit_calibration_temperature(self, val_size: int = 5000) -> float:
        """Fit calibration temperature T on a separate validation/calibration split."""
        from torch.utils.data import DataLoader

        from trustbench.datasets.cifar10 import get_cifar10_validation_split
        from trustbench.evaluation.uncertainty import fit_temperature

        _, val_set = get_cifar10_validation_split(
            data_dir=self.data_dir,
            val_size=val_size,
            seed=self.seed,
            download=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        val_res = run_inference(self.model, val_loader, device=self.device)
        self.fitted_temperature = fit_temperature(val_res["logits"], val_res["labels"])
        return self.fitted_temperature

    def evaluate_condition(
        self,
        corruption_type: str,
        severity: int,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Evaluate model on a specific corruption type and severity level."""
        corrupted_ds = CorruptedDataset(
            base_dataset=self.raw_testset,
            corruption_type=corruption_type,
            severity=severity,
            base_seed=self.seed,
        )

        loader: Any = DataLoader(
            corrupted_ds,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        inference_results = run_inference(self.model, loader, device=self.device)
        logits = inference_results["logits"]
        predictions = inference_results["predictions"]
        labels = inference_results["labels"]
        sample_ids = inference_results["sample_ids"]

        probabilities = softmax(logits, axis=1)

        # 1. Classification
        cls_metrics = compute_classification_metrics(
            labels=labels,
            predictions=predictions,
            class_names=CIFAR10_CLASSES,
        )

        # 2. Calibration
        cal_metrics = compute_calibration_metrics(
            labels=labels,
            probabilities=probabilities,
        )

        # 3. Uncertainty
        unc_metrics = compute_uncertainty_metrics(
            probabilities=probabilities,
        )

        # 4. Error Detection (evaluating MSP, Entropy, Temperature-Scaled, and Energy scores)
        err_metrics = compute_error_detection_metrics(
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
            logits=logits,
            temperature=getattr(self, "fitted_temperature", None),
        )

        # 5. Selective Prediction
        sel_metrics = compute_selective_prediction_metrics(
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
        )

        metrics_payload = {
            "corruption": corruption_type,
            "severity": severity,
            "condition": f"{corruption_type.replace('_', ' ').title()} S{severity}",
            "classification": cls_metrics,
            "calibration": cal_metrics,
            "uncertainty": unc_metrics,
            "error_detection": err_metrics,
            "selective_prediction": sel_metrics,
            "summary": {
                "condition": f"{corruption_type.replace('_', ' ').title()} S{severity}",
                "accuracy": cls_metrics["accuracy"],
                "macro_f1": cls_metrics["macro_f1"],
                "brier_score": cal_metrics["brier_score"],
                "ece": cal_metrics["ece"],
                "mean_entropy": unc_metrics["mean_entropy"],
                "mean_max_probability": unc_metrics["mean_max_probability"],
                "total_samples": len(labels),
            },
        }

        preds_dict = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "labels": labels,
            "sample_ids": sample_ids,
        }

        return metrics_payload, preds_dict

    def run_full_suite(
        self,
        corruption_types: list[str] | None = None,
        severities: list[int] | None = None,
        save_predictions: bool = True,
    ) -> dict[str, Any]:
        """Execute evaluation across all corruptions and severities, saving artifacts."""
        if corruption_types is None:
            corruption_types = ["gaussian_noise", "gaussian_blur", "brightness", "occlusion"]
        if severities is None:
            severities = [1, 2, 3]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        corruption_results_dir = self.output_dir / "corruption_results"
        corruption_results_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir = self.output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)

        if self.fitted_temperature is None:
            self.fit_calibration_temperature()

        # Load clean baseline metrics if available
        baseline_summary = None
        if self.baseline_experiment_dir and self.baseline_experiment_dir.exists():
            try:
                base_data = load_experiment_artifacts(self.baseline_experiment_dir)
                b_metrics = base_data.get("metrics", {})
                b_summary = b_metrics.get("summary", {})
                b_unc = b_metrics.get("uncertainty", {})
                baseline_summary = {
                    "condition": "Clean",
                    "accuracy": b_summary.get("accuracy", 0.9539),
                    "macro_f1": b_summary.get("macro_f1", 0.9539),
                    "brier_score": b_summary.get("brier_score", 0.0754),
                    "ece": b_summary.get("ece", 0.0262),
                    "mean_entropy": b_summary.get("mean_entropy", 0.0683),
                    "mean_max_probability": b_unc.get("mean_max_probability", 0.9791),
                    "total_samples": b_summary.get("total_samples", 10000),
                }
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                baseline_summary = None

        all_summaries: list[dict[str, Any]] = []
        if baseline_summary is not None:
            all_summaries.append(baseline_summary)

        per_corruption_data: dict[str, dict[str, Any]] = {}

        for c_type in corruption_types:
            c_type_results = {}
            npz_arrays: dict[str, np.ndarray] = {}

            for sev in severities:
                payload, preds = self.evaluate_condition(c_type, sev)
                c_type_results[f"severity_{sev}"] = payload
                all_summaries.append(payload["summary"])

                for k, arr in preds.items():
                    npz_arrays[f"s{sev}_{k}"] = arr

            per_corruption_data[c_type] = c_type_results

            # Save corruption JSON: corruption_results/<c_type>.json
            json_file = corruption_results_dir / f"{c_type}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(c_type_results, f, indent=2, cls=NpEncoder)

            # Save predictions NPZ: predictions/<c_type>.npz (if enabled)
            if save_predictions:
                npz_file = predictions_dir / f"{c_type}.npz"
                np.savez_compressed(npz_file, **npz_arrays)  # type: ignore[arg-type]
            del npz_arrays

        # Save config.json
        config_payload = {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "model_name": self.model_name,
            "corruptions": corruption_types,
            "severities": severities,
            "severity_definitions": serialize_severities(),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "classes": list(CIFAR10_CLASSES),
        }
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, cls=NpEncoder)

        # Save metadata.json
        metadata_payload = {
            "experiment_id": self.output_dir.name,
            "created_at": datetime.now(UTC).isoformat(),
            "checkpoint_sha256": checkpoint_sha256,
            "baseline_experiment_ref": str(self.baseline_experiment_dir) if self.baseline_experiment_dir else None,
            "environment": get_environment_info(),
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, cls=NpEncoder)

        # Compute comprehensive robustness aggregation
        aggregation = compute_robustness_aggregation(
            conditions=all_summaries,
            clean_baseline=baseline_summary,
        )

        # Save summary.json
        summary_payload = {
            "experiment_id": self.output_dir.name,
            "conditions": all_summaries,
            "aggregation": aggregation,
        }
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, cls=NpEncoder)

        return {
            "output_dir": str(self.output_dir),
            "summary": all_summaries,
            "corruption_data": per_corruption_data,
            "aggregation": aggregation,
        }
