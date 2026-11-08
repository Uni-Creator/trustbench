# Robustness & Input Corruptions

TrustBench benchmarks model stability under common physical, environmental, and digital input corruptions across progressive severity levels (1 to 5).

---

## Supported Corruptions

| Corruption Type | Category | Description |
|---|---|---|
| `gaussian_noise` | Noise | Zero-mean Gaussian perturbation with escalating standard deviations |
| `shot_noise` | Noise | Poisson noise mimicking low-light photographic sensor noise |
| `impulse_noise` | Noise | Salt-and-pepper bit transmission dropouts |
| `gaussian_blur` | Blur | Gaussian filtering with increasing kernel radius and sigma |
| `defocus_blur` | Blur | Disk kernel convolution simulating optical defocusing |
| `motion_blur` | Blur | Directional blur simulating camera or subject movement |
| `brightness` | Environmental | Scaled pixel intensities mimicking over/underexposure |
| `contrast` | Environmental | Dynamic range compression toward mean luminance |
| `fog` | Environmental | Diamond-square atmospheric fog simulation |
| `jpeg_compression`| Digital | Discrete Cosine Transform (DCT) lossy block compression |
| `pixelate` | Digital | Resolution downsampling and nearest-neighbor upscaling |

---

## Evaluating Robustness

### Using the Python API
```python
from trustbench.robustness.evaluator import RobustnessEvaluator

evaluator = RobustnessEvaluator(
    checkpoint_path="checkpoint/ckpt.pth",
    model_name="ResNet18",
    data_dir="dataset",
    output_dir="artifacts/experiments/robustness_run",
    batch_size=128,
)

# Evaluate Gaussian noise at severity 1
metrics, predictions = evaluator.evaluate_condition("gaussian_noise", severity=1)
print(f"Accuracy: {metrics['classification']['accuracy']:.2%}")
```

### Using Configuration (`trustbench.yaml`)
```yaml
robustness:
  enabled: true
  corruption_types:
    - gaussian_noise
    - gaussian_blur
    - brightness
  severity_levels: [1, 2, 3, 4, 5]
```
Running `trustbench evaluate trustbench.yaml` automatically benchmarks all combinations and generates robustness degradation plots.
