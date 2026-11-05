"""Unit tests for Phase 3 uncertainty methods: MC Dropout, Ensembles, and Energy Scores."""

import numpy as np
import torch
from torch import nn

from trustbench.evaluation.uncertainty import (
    apply_temperature_scaling,
    compute_energy_scores,
    compute_ensemble_uncertainties,
    compute_uncertainty_metrics,
    evaluate_deep_ensemble,
    evaluate_mc_dropout,
    fit_temperature,
)


def test_energy_scores():
    # Logits: [10.0, 0.0] vs [0.0, 0.0]
    logits = np.array([[10.0, 0.0], [0.0, 0.0]])
    energy = compute_energy_scores(logits, temperature=1.0)
    # Energy = -logsumexp(logits)
    # Higher logit norm means lower energy (higher certainty)
    assert energy[0] < energy[1]


def test_fit_and_apply_temperature():
    val_logits = np.array([[5.0, 1.0], [1.0, 5.0], [4.0, 2.0], [2.0, 4.0]])
    val_labels = np.array([0, 1, 0, 1])

    T = fit_temperature(val_logits, val_labels)
    assert T > 0.0

    cal_probs, cal_msp = apply_temperature_scaling(val_logits, T)
    assert cal_probs.shape == val_logits.shape
    assert np.allclose(np.sum(cal_probs, axis=1), 1.0)
    assert np.allclose(cal_msp, np.max(cal_probs, axis=1))


def test_ensemble_uncertainty_decomposition():
    # 3 ensemble members, 2 samples, 2 classes
    # Sample 0: all members agree on [0.9, 0.1] -> low epistemic uncertainty
    # Sample 1: member 0 says [0.9, 0.1], member 1 says [0.1, 0.9], member 2 says [0.5, 0.5] -> high epistemic
    member_probs = np.array([
        [[0.9, 0.1], [0.9, 0.1]],
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.9, 0.1], [0.5, 0.5]],
    ])

    decomp = compute_ensemble_uncertainties(member_probs)

    assert "total_uncertainty" in decomp
    assert "aleatoric_uncertainty" in decomp
    assert "epistemic_uncertainty" in decomp
    assert "predictive_variance" in decomp

    # Mathematical property: Total >= Aleatoric
    assert np.all(decomp["total_uncertainty"] >= decomp["aleatoric_uncertainty"] - 1e-6)

    # Sample 1 has higher epistemic uncertainty than sample 0
    assert decomp["epistemic_uncertainty"][1] > decomp["epistemic_uncertainty"][0]
    # Sample 1 has higher predictive variance than sample 0
    assert decomp["predictive_variance"][1] > decomp["predictive_variance"][0]


class SimpleDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.fc(self.drop(x))


def test_evaluate_mc_dropout():
    torch.manual_seed(42)
    model = SimpleDropoutModel()
    inputs = torch.randn(5, 4)

    mc_res = evaluate_mc_dropout(model, inputs, num_samples=10)
    assert mc_res["mean_probabilities"].shape == (5, 2)
    assert len(mc_res["total_uncertainty"]) == 5
    assert len(mc_res["epistemic_uncertainty"]) == 5
    assert mc_res["summary"]["mean_total_uncertainty"] >= 0.0


def test_evaluate_deep_ensemble():
    torch.manual_seed(42)
    models = [nn.Linear(4, 2) for _ in range(3)]
    inputs = torch.randn(5, 4)

    ens_res = evaluate_deep_ensemble(models, inputs)
    assert ens_res["mean_probabilities"].shape == (5, 2)
    assert len(ens_res["total_uncertainty"]) == 5
    assert len(ens_res["epistemic_uncertainty"]) == 5


def test_uncertainty_metrics_with_logits():
    probs = np.array([[0.8, 0.2], [0.5, 0.5]])
    logits = np.array([[2.0, 0.5], [1.0, 1.0]])

    summary = compute_uncertainty_metrics(probs, logits=logits)
    assert "mean_entropy" in summary
    assert "mean_max_probability" in summary
    assert "mean_energy" in summary
