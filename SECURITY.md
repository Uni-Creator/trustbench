# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2.0 | :x:                |

## Security Architecture & Best Practices

TrustBench adopts a secure-by-default posture:
1. **PyTorch Checkpoint Loading**: By default, `load_model_from_checkpoint()` enforces `weights_only=True` to prevent arbitrary Python code execution via malicious pickle files.
2. **Archive Extraction Protection**: Tarball extraction (`import_experiment()`) validates member paths to prevent directory traversal / ZipSlip attacks (`SecurityError`).
3. **YAML Parsing**: All configuration loading uses `yaml.safe_load()`.
4. **Plugin Discovery**: TrustBench discovers plugins strictly via standard Python packaging entry points (`trustbench.plugins`). Arbitrary script directory execution is disabled.

## Reporting a Vulnerability

If you discover a potential security vulnerability in TrustBench, please report it privately:

- **Email**: Send vulnerability reports directly to `abhayr24564@gmail.com`.
- **Subject**: `[SECURITY VULNERABILITY] TrustBench - <Brief Summary>`
- **Content**: Please include:
  - Description of the vulnerability and attack vector
  - Step-by-step reproduction instructions or proof-of-concept
  - Affected versions and configurations

We will acknowledge receipt within 48 hours and work with you on a resolution prior to public disclosure.
