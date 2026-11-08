# Contributing to TrustBench

Thank you for your interest in contributing to TrustBench! We welcome contributions from researchers, engineers, and community members.

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Uni-Creator/trustbench.git
   cd trustbench
   ```

2. **Install with development dependencies**:
   TrustBench uses [uv](https://github.com/astral-sh/uv) or standard pip/virtualenv:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[all]"
   ```

3. **Verify the test suite**:
   ```bash
   pytest
   ```

4. **Verify linting**:
   ```bash
   ruff check .
   ```

## Pull Request Guidelines

1. **Create a topic branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Keep the core small and framework-agnostic**:
   - Never import optional frameworks (`torch`, `tensorflow`, `jax`, `onnxruntime`, `plotly`) at top-level core modules (`trustbench.core`).
   - Use dynamic imports and verify with import isolation tests.
3. **Add unit tests**:
   - Place tests under `tests/unit/` or `tests/integration/`.
   - Ensure `pytest` passes with 100% clean exit.
4. **Adhere to Code Style**:
   - Format and lint using `ruff check --fix .`.
5. **Update Documentation**:
   - If adding a new metric, adapter, or configuration parameter, update relevant documents in `docs/` and `README.md`.
6. **Submit PR**:
   - Provide a clear summary of changes, rationale, and test results using the pull request template.

## Reporting Bugs

Please use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md) and include:
- Operating system and Python version
- TrustBench version (`trustbench version`)
- Minimal reproducible example
- Full stack trace
