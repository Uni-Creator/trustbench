"""Performance benchmarking script for TrustBench (Phase 7).

Measures:
1. Dataset loading time
2. Inference duration & throughput (samples/sec) across batch sizes
3. CPU memory (tracemalloc peak)
4. GPU memory (torch.cuda.max_memory_allocated peak)
5. Metric computation duration
6. Artifact serialization time

Evaluates representative small (500), medium (5,000), and large (20,000) datasets.
"""

from __future__ import annotations

import gc
import json
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from trustbench.core.evaluator import evaluate
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.models.pytorch import PyTorchModelAdapter
from trustbench.reporting.artifacts import save_experiment_artifacts
from trustbench.reporting.html_report import generate_html_report
from trustbench.reporting.schema import generate_json_report


class SimpleConvNet(nn.Module):
    """Representative small CNN for reproducible benchmarks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)


def generate_benchmark_data(
    n_samples: int,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Generate synthetic image data (N, 3, 32, 32) and labels."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    images = torch.randn(n_samples, 3, 32, 32, dtype=torch.float32)
    labels = rng.integers(0, num_classes, size=n_samples).astype(np.int64)
    sample_ids = np.arange(n_samples)
    return images, labels, sample_ids


def measure_dataset_loading(n_samples: int) -> tuple[float, InMemoryDatasetAdapter]:
    """Measure time to initialize and construct InMemoryDatasetAdapter."""
    t0 = time.perf_counter()
    images, labels, sample_ids = generate_benchmark_data(n_samples)
    adapter = InMemoryDatasetAdapter(inputs=images, labels=labels, sample_ids=sample_ids)
    dt = time.perf_counter() - t0
    return dt, adapter


def benchmark_inference(
    model_adapter: PyTorchModelAdapter,
    dataset_adapter: InMemoryDatasetAdapter,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Measure inference throughput, latency, CPU memory, and GPU memory."""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    t0 = time.perf_counter()

    from trustbench.core.inference import run_adapter_inference

    output = run_adapter_inference(model_adapter, dataset_adapter, batch_size=batch_size)

    duration = time.perf_counter() - t0
    _current_mem, peak_cpu_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_gpu_mem = 0.0
    if device == "cuda" and torch.cuda.is_available():
        peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

    n_samples = len(dataset_adapter)
    throughput = n_samples / duration if duration > 0 else 0.0

    return {
        "batch_size": batch_size,
        "n_samples": n_samples,
        "duration_sec": duration,
        "throughput_samples_per_sec": throughput,
        "peak_cpu_memory_mb": peak_cpu_mem / (1024 * 1024),
        "peak_gpu_memory_mb": peak_gpu_mem,
        "output": output,
    }


def benchmark_metrics_computation(
    model_adapter: PyTorchModelAdapter,
    dataset_adapter: InMemoryDatasetAdapter,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Measure evaluation metric computation time."""
    t0 = time.perf_counter()
    result = evaluate(
        model_adapter,
        dataset_adapter,
        metrics=("classification", "calibration", "uncertainty", "error_detection", "selective_prediction"),
        batch_size=batch_size,
    )
    duration = time.perf_counter() - t0
    return {
        "duration_sec": duration,
        "result": result,
    }


def benchmark_artifact_serialization(result: Any, n_samples: int) -> dict[str, Any]:
    """Measure time to serialize artifacts (NPZ, JSON, HTML)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Standard artifacts
        t0 = time.perf_counter()
        save_experiment_artifacts(
            output_dir=tmp_path / "artifacts",
            metadata={"n_samples": n_samples},
            metrics=result.metrics,
            config={},
            predictions=result.to_predictions_dict(),
        )
        std_time = time.perf_counter() - t0

        # 2. JSON report
        t1 = time.perf_counter()
        generate_json_report(result, tmp_path / "artifacts")
        json_time = time.perf_counter() - t1

        # 3. HTML report
        t2 = time.perf_counter()
        generate_html_report(result, tmp_path / "artifacts")
        html_time = time.perf_counter() - t2

        total_time = time.perf_counter() - t0

        npz_size_mb = 0.0
        npz_file = tmp_path / "artifacts" / "predictions.npz"
        if npz_file.exists():
            npz_size_mb = npz_file.stat().st_size / (1024 * 1024)

    return {
        "standard_artifacts_sec": std_time,
        "json_report_sec": json_time,
        "html_report_sec": html_time,
        "total_serialization_sec": total_time,
        "predictions_npz_size_mb": npz_size_mb,
    }


def run_full_benchmark() -> dict[str, Any]:
    """Run full benchmark across small (500), medium (5,000), and large (20,000) datasets."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== TrustBench Performance Benchmark ===")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_sizes = [500, 5000, 20000]
    batch_sizes = [32, 64, 128, 256]

    results: dict[str, Any] = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "benchmarks": {},
    }

    model = SimpleConvNet(num_classes=10)
    model_adapter = PyTorchModelAdapter(model, num_classes=10, device=device)

    for n_samples in dataset_sizes:
        size_label = f"size_{n_samples}"
        print(f"\n--- Benchmarking N = {n_samples:,} samples ---")

        # 1. Dataset loading
        ds_load_time, ds_adapter = measure_dataset_loading(n_samples)
        print(f"  Dataset generation & loading: {ds_load_time:.4f}s")

        size_res: dict[str, Any] = {
            "n_samples": n_samples,
            "dataset_loading_sec": ds_load_time,
            "inference_batch_tuning": [],
        }

        # 2. Batch size tuning for inference
        for bs in batch_sizes:
            inf_res = benchmark_inference(model_adapter, ds_adapter, batch_size=bs, device=device)
            print(
                f"  Batch size {bs:3d}: {inf_res['throughput_samples_per_sec']:8.1f} samples/s | "
                f"Latency: {inf_res['duration_sec']:.3f}s | "
                f"CPU Peak: {inf_res['peak_cpu_memory_mb']:.2f} MB | "
                f"GPU Peak: {inf_res['peak_gpu_memory_mb']:.2f} MB"
            )
            # Remove heavy array output before saving results
            del inf_res["output"]
            size_res["inference_batch_tuning"].append(inf_res)

        # 3. Metrics computation (using best default bs=128)
        metric_res = benchmark_metrics_computation(model_adapter, ds_adapter, batch_size=128)
        print(f"  Metric evaluation time (all 5 groups): {metric_res['duration_sec']:.4f}s")
        size_res["metric_computation_sec"] = metric_res["duration_sec"]

        # 4. Serialization
        eval_result = metric_res["result"]
        ser_res = benchmark_artifact_serialization(eval_result, n_samples)
        print(
            f"  Serialization: total {ser_res['total_serialization_sec']:.4f}s "
            f"(NPZ+JSON: {ser_res['standard_artifacts_sec']:.4f}s, "
            f"HTML: {ser_res['html_report_sec']:.4f}s, "
            f"NPZ size: {ser_res['predictions_npz_size_mb']:.2f} MB)"
        )
        size_res["serialization"] = ser_res

        results["benchmarks"][size_label] = size_res

    return results


if __name__ == "__main__":
    benchmark_data = run_full_benchmark()
    out_path = Path("benchmarks/results_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n✓ Saved baseline benchmark results to {out_path}")
