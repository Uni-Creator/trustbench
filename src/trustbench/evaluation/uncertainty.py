"""Uncertainty quantification methods: MSP, Entropy, Temperature Scaling, Energy scores, MC Dropout, and Ensembles.

Note on Terminology:
  - Confidence Score: Higher means model is more certain/confident (e.g. Maximum Softmax Probability).
  - Predictive Uncertainty: Higher means model is less certain (e.g. Entropy, Energy Score, Variance).
  - Aleatoric Uncertainty: Data/observation noise (inherent ambiguity).
  - Epistemic Uncertainty: Model/knowledge uncertainty (can be reduced with more training data).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def compute_predictive_entropy(
    probabilities: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute per-sample Shannon entropy: H(p) = -sum_c (p_c * log(p_c)).

    Entropy measures total predictive uncertainty. Minimum (0.0) indicates complete certainty;
    maximum (log(C)) indicates uniform uncertainty over C classes.

    Args:
        probabilities: 2D array of class probabilities of shape (N, C).
        epsilon: Small epsilon floor to prevent log(0).

    Returns:
        1D array of entropy values of shape (N,).
    """
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (N, C).")

    clipped_probs = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)
    return entropy


def compute_energy_scores(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute negative free energy score: E(x; T) = -T * logsumexp(logits / T).

    In energy-based OOD/uncertainty detection, lower energy represents higher model certainty,
    meaning uncertainty is directly measured by positive free energy: -E(x) = T * logsumexp(logits / T).

    Args:
        logits: 2D array of raw output logits of shape (N, C).
        temperature: Temperature scaling factor.

    Returns:
        1D array of energy scores of shape (N,).
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array of shape (N, C).")
    energy = -temperature * logsumexp(logits / temperature, axis=1)
    return energy


def fit_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """Fit optimal temperature scalar T on a held-out validation/calibration split using NLL.

    Args:
        val_logits: 2D array of validation logits of shape (N_val, C).
        val_labels: 1D array of validation labels of shape (N_val,).

    Returns:
        Optimal temperature T > 0.
    """
    val_logits = np.asarray(val_logits, dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)
    n_samples, _n_classes = val_logits.shape

    def nll_objective(t: np.ndarray) -> float:
        temp = float(t[0])
        scaled_logits = val_logits / temp
        log_probs = scaled_logits - logsumexp(scaled_logits, axis=1, keepdims=True)
        loss = -np.mean(log_probs[np.arange(n_samples), val_labels])
        return float(loss)

    res = minimize(
        nll_objective,
        x0=np.array([1.5]),
        bounds=[(0.01, 10.0)],
        method="L-BFGS-B",
    )
    return float(res.x[0])


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply temperature scaling to logits and compute calibrated probabilities and max confidence.

    Args:
        logits: 2D array of shape (N, C).
        temperature: Scaling temperature T.

    Returns:
        Tuple of (calibrated_probabilities of shape (N, C), calibrated_max_probabilities of shape (N,)).
    """
    logits = np.asarray(logits, dtype=float)
    scaled_logits = logits / temperature
    cal_probs = softmax(scaled_logits, axis=1)
    cal_msp = np.max(cal_probs, axis=1)
    return cal_probs, cal_msp


def compute_uncertainty_metrics(
    probabilities: np.ndarray,
    logits: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute summary metrics for prediction uncertainty and confidence.

    Distinguishes:
      - Predictive Uncertainty: Shannon entropy, energy score.
      - Confidence: Maximum Softmax Probability (MSP).

    Args:
        probabilities: 2D array of class probabilities (N, C).
        logits: Optional 2D array of unscaled output logits (N, C).
        epsilon: Numerical floor for entropy computation.

    Returns:
        Dictionary of uncertainty and confidence summary statistics.
    """
    probabilities = np.asarray(probabilities)
    if len(probabilities) == 0:
        return {
            "mean_entropy": 0.0,
            "std_entropy": 0.0,
            "min_entropy": 0.0,
            "max_entropy": 0.0,
            "median_entropy": 0.0,
            "mean_max_probability": 0.0,
        }

    entropies = compute_predictive_entropy(probabilities, epsilon=epsilon)
    max_probs = np.max(probabilities, axis=1)

    result: dict[str, Any] = {
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "min_entropy": float(np.min(entropies)),
        "max_entropy": float(np.max(entropies)),
        "median_entropy": float(np.median(entropies)),
        "mean_max_probability": float(np.mean(max_probs)),
    }

    if logits is not None:
        energy = compute_energy_scores(logits)
        result["mean_energy"] = float(np.mean(energy))
        result["std_energy"] = float(np.std(energy))

    return result


def compute_ensemble_uncertainties(
    member_probabilities: np.ndarray | Sequence[np.ndarray],
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Decompose predictive uncertainty from ensemble / MC samples into aleatoric and epistemic components.

    Mathematical formulation (Information-Theoretic Decomposition):
      - Total Uncertainty = H( E[p] )  [Predictive Entropy of the ensemble mean]
      - Aleatoric Uncertainty = E[ H(p) ]  [Average entropy across members/samples]
      - Epistemic Uncertainty = Total - Aleatoric = Mutual Information I(Y; θ | x)
      - Predictive Variance = (1 / C) * sum_c Var_k(p_{k, c})

    Args:
        member_probabilities: Array of shape (K, N, C) where K is members/samples,
            or sequence of K arrays of shape (N, C).
        epsilon: Numerical floor for entropy calculation.

    Returns:
        Dictionary with per-sample arrays and summary metrics for:
            - 'mean_probabilities': (N, C) ensemble averaged probabilities
            - 'total_uncertainty': (N,) total entropy
            - 'aleatoric_uncertainty': (N,) expected data uncertainty
            - 'epistemic_uncertainty': (N,) mutual information / knowledge uncertainty
            - 'predictive_variance': (N,) mean variance across classes
            - 'summary': scalar statistics
    """
    if isinstance(member_probabilities, (list, tuple)):
        stacked = np.stack([np.asarray(p) for p in member_probabilities], axis=0)
    else:
        stacked = np.asarray(member_probabilities)

    if stacked.ndim != 3:
        raise ValueError("member_probabilities must have shape (K, N, C).")

    _k_members, n_samples, _n_classes = stacked.shape
    if n_samples == 0:
        return {
            "mean_probabilities": np.empty((0, stacked.shape[2])),
            "total_uncertainty": np.empty(0),
            "aleatoric_uncertainty": np.empty(0),
            "epistemic_uncertainty": np.empty(0),
            "predictive_variance": np.empty(0),
            "summary": {
                "mean_total_uncertainty": 0.0,
                "mean_aleatoric": 0.0,
                "mean_epistemic": 0.0,
                "mean_variance": 0.0,
            },
        }

    # Mean ensemble probability: (N, C)
    mean_probs = np.mean(stacked, axis=0)

    # Total uncertainty: H(mean_probs)
    total_entropy = compute_predictive_entropy(mean_probs, epsilon=epsilon)

    # Aleatoric uncertainty: mean of member entropies
    member_entropies = np.array([compute_predictive_entropy(stacked[k], epsilon=epsilon) for k in range(_k_members)])
    aleatoric = np.mean(member_entropies, axis=0)

    # Epistemic uncertainty: Mutual Information = Total - Aleatoric (clipped at 0.0 for numerical safety)
    epistemic = np.maximum(0.0, total_entropy - aleatoric)

    # Predictive variance: variance across members averaged over classes
    var_per_class = np.var(stacked, axis=0)  # shape (N, C)
    mean_variance = np.mean(var_per_class, axis=1)  # shape (N,)

    return {
        "mean_probabilities": mean_probs,
        "total_uncertainty": total_entropy,
        "aleatoric_uncertainty": aleatoric,
        "epistemic_uncertainty": epistemic,
        "predictive_variance": mean_variance,
        "summary": {
            "mean_total_uncertainty": float(np.mean(total_entropy)),
            "mean_aleatoric": float(np.mean(aleatoric)),
            "mean_epistemic": float(np.mean(epistemic)),
            "mean_variance": float(np.mean(mean_variance)),
        },
    }


def evaluate_mc_dropout(
    model: Any,
    inputs: Any,
    num_samples: int = 10,
    device: Any | None = None,
) -> dict[str, Any]:
    """Perform Monte Carlo Dropout evaluation by executing multiple stochastic forward passes.

    Enables dropout at inference time to obtain an empirical distribution over network predictions.

    Args:
        model: PyTorch nn.Module with dropout layers.
        inputs: Input tensor of shape (N, ...) or PyTorch DataLoader.
        num_samples: Number of stochastic forward passes (default: 10).
        device: Torch device (CPU or CUDA).

    Returns:
        Dictionary from `compute_ensemble_uncertainties`.
    """
    import torch

    model.train()  # Enable dropout layers
    stochastic_probs: list[np.ndarray] = []

    with torch.no_grad():
        for _ in range(num_samples):
            if isinstance(inputs, torch.Tensor):
                in_tensor = inputs.to(device) if device is not None else inputs
                out = model(in_tensor)
                logits = out.detach().cpu().numpy()
                probs = softmax(logits, axis=1)
                stochastic_probs.append(probs)
            else:
                # DataLoader
                batch_probs = []
                for batch in inputs:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    if device is not None:
                        x = x.to(device)
                    out = model(x)
                    lgt = out.detach().cpu().numpy()
                    batch_probs.append(softmax(lgt, axis=1))
                stochastic_probs.append(np.concatenate(batch_probs, axis=0))

    return compute_ensemble_uncertainties(stochastic_probs)


def evaluate_deep_ensemble(
    models: Sequence[Any],
    inputs: Any,
    device: Any | None = None,
) -> dict[str, Any]:
    """Evaluate an ensemble of diverse models to quantify predictive and epistemic uncertainty.

    Args:
        models: Sequence of model instances or adapters.
        inputs: Input tensor or DataLoader.
        device: Torch device.

    Returns:
        Dictionary from `compute_ensemble_uncertainties`.
    """
    import torch

    member_probs: list[np.ndarray] = []

    with torch.no_grad():
        for m in models:
            if hasattr(m, "eval"):
                m.eval()
            if isinstance(inputs, torch.Tensor):
                in_tensor = inputs.to(device) if device is not None else inputs
                out = m(in_tensor)
                logits = out.detach().cpu().numpy()
                probs = softmax(logits, axis=1)
                member_probs.append(probs)
            elif hasattr(m, "predict_proba"):
                probs = m.predict_proba(inputs)
                member_probs.append(probs)
            else:
                batch_probs = []
                for batch in inputs:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    if device is not None and hasattr(x, "to"):
                        x = x.to(device)
                    out = m(x)
                    lgt = out.detach().cpu().numpy()
                    batch_probs.append(softmax(lgt, axis=1))
                member_probs.append(np.concatenate(batch_probs, axis=0))

    return compute_ensemble_uncertainties(member_probs)
