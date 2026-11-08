# Contributing Guide

Please refer to the root [CONTRIBUTING.md](../CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) files for guidelines on how to contribute to TrustBench.

## Key Developer Commands

```bash
# Run test suite
pytest

# Run linter and formatting checks
ruff check .

# Run static type checking
mypy src/trustbench/core src/trustbench/reporting src/trustbench/metrics

# Build distribution wheels
uv build
```
