"""ONNX Runtime model adapter for framework-agnostic inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from trustbench.core.exceptions import ShapeError, UnsupportedCapabilityError
from trustbench.models.base import ModelAdapter


class ONNXModelAdapter(ModelAdapter):
    """Adapter for ONNX models executed via ONNX Runtime (``onnxruntime``).

    Supports:
      - File paths to ``.onnx`` files or existing ``InferenceSession`` instances.
      - Input tensor feeding by input name.
      - Logits and probability extraction with softmax conversion.
    """

    def __init__(
        self,
        model_or_path: str | Path | Any,
        num_classes: int,
        input_name: str | None = None,
        output_name: str | None = None,
        providers: list[str] | None = None,
        task: str = "classification",
    ) -> None:
        """Initialize ONNX adapter.

        Args:
            model_or_path: Path to `.onnx` model or an `onnxruntime.InferenceSession`.
            num_classes: Number of output classes (or 1 for regression).
            input_name: Optional explicit name of input tensor. If None, uses first input.
            output_name: Optional explicit name of output tensor. If None, uses first output.
            providers: List of execution providers (e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']).
            task: Task type ('classification' or 'regression').
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX Runtime is required for ONNXModelAdapter. "
                "Install it with: pip install onnxruntime (or onnxruntime-gpu)"
            ) from exc

        if isinstance(model_or_path, (str, Path)):
            model_path_str = str(model_or_path)
            if providers is None:
                providers = ort.get_available_providers()
            self._session = ort.InferenceSession(model_path_str, providers=providers)
        elif hasattr(model_or_path, "run"):
            self._session = model_or_path
        else:
            raise TypeError(
                f"Expected str, Path, or InferenceSession, got {type(model_or_path).__name__}."
            )

        self._num_classes = int(num_classes)
        self._task = task.lower()

        # Discover input/output names
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()

        self._input_name = input_name or inputs[0].name
        self._output_name = output_name or outputs[0].name

    @property
    def session(self) -> Any:
        """Underlying ONNX Runtime InferenceSession."""
        return self._session

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def supports_predict_proba(self) -> bool:
        return self._task == "classification"

    @property
    def supports_predict_logits(self) -> bool:
        return True

    def _run_session(self, inputs: Any) -> np.ndarray:
        """Run ONNX session on given input batch."""
        arr = np.asarray(inputs)
        # Match input tensor dtype if expected is float32
        input_meta = next(
            (i for i in self._session.get_inputs() if i.name == self._input_name), None
        )
        if input_meta and "float" in input_meta.type and arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        raw_outputs = self._session.run([self._output_name], {self._input_name: arr})
        out = np.asarray(raw_outputs[0])
        return out

    def predict(self, inputs: Any) -> np.ndarray:
        """Return predicted class indices (classification) or values (regression)."""
        out = self._run_session(inputs)

        if self._task == "classification":
            if out.ndim >= 2 and out.shape[1] == self._num_classes:
                return np.argmax(out, axis=1)
            if out.ndim == 1:
                return (out > 0.5).astype(int) if self._num_classes == 2 else out.astype(int)
            return np.argmax(out, axis=-1)

        return out.squeeze()

    def predict_logits(self, inputs: Any) -> np.ndarray:
        """Return raw model outputs."""
        out = self._run_session(inputs)
        if out.ndim == 1 and self._num_classes == 2:
            out = np.column_stack([-out, out])
        return out

    def predict_proba(self, inputs: Any) -> np.ndarray:
        """Return softmax probabilities."""
        if self._task != "classification":
            raise UnsupportedCapabilityError("predict_proba is only available for classification.")

        logits = self.predict_logits(inputs)
        if logits.ndim != 2:
            raise ShapeError(f"Expected 2D logits for predict_proba, got shape {logits.shape}.")

        # Softmax with numerical stability
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
