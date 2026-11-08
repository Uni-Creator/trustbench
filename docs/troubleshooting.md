# Troubleshooting & FAQ

Common questions, error resolutions, and debugging tips for TrustBench.

---

## 1. Installation Issues

### `ModuleNotFoundError: No module named 'torch'`
**Cause**: TrustBench was installed without the PyTorch extra.
**Fix**: Install PyTorch support with:
```bash
pip install "trustbench[pytorch]"
```

### `ImportError: ONNX Runtime is required for ONNXModelAdapter`
**Cause**: The ONNX extra is missing.
**Fix**: Install with:
```bash
pip install "trustbench[onnx]"
```

---

## 2. Configuration & Execution Issues

### `ValidationError: checkpoint_path does not exist`
**Cause**: Relative paths in `trustbench.yaml` are resolved relative to the directory containing `trustbench.yaml`.
**Fix**: Ensure your path matches the location relative to the config file:
```yaml
model:
  checkpoint_path: checkpoint/ckpt.pth
```

### `ShapeError: Expected probabilities of shape (N, C)`
**Cause**: Your model's `predict_proba()` emitted an unexpected dimension or number of classes.
**Fix**: Verify that `num_classes` matches the number of columns returned by your model.

---

## 3. Hardware & GPU Placement

### CUDA Out of Memory (OOM) during Evaluation
**Fix**: Reduce `batch_size` in your `trustbench.yaml` or constructor:
```yaml
dataset:
  batch_size: 64 # Reduce from 128 or 256
```
TrustBench uses pre-allocated memory buffers to minimize peak RAM, but GPU memory depends on model size and image resolution.

### Forcing CPU Execution
To evaluate on CPU even when CUDA is present:
```yaml
model:
  device: cpu
```
Or in Python:
```python
adapter = PyTorchModelAdapter(model, num_classes=10, device="cpu")
```
