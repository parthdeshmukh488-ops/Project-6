"""Error-mitigation tests: folding, extrapolation, readout correction."""

import numpy as np
import pytest
from qvolve import Circuit

from vqelab import fermionic_hamiltonian, load_molecule
from vqelab.ansatz import UCCSDAnsatz
from vqelab.estimator import StatevectorEstimator
from vqelab.mitigation import (
    ReadoutMitigator,
    fold_global,
    zero_noise_extrapolation,
)
from vqelab.noise import DepolarizingNoise, TrajectoryEstimator


# ----------------------------------------------------------------------
# unitary folding
# ----------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1, 3, 5, 7])
def test_folding_preserves_the_noiseless_state(scale):
    """C (C^dag C)^n is the identity, so the ideal answer must not move.

    If this failed, every extrapolation would be measuring the wrong thing.
    """
    circuit = Circuit.random_circuit(4, 5, seed=0)
    original = circuit.run(fuse=False).statevector.data
    folded = fold_global(circuit, scale).run(fuse=False).statevector.data
    assert np.allclose(folded, original, atol=1e-11)


@pytest.mark.parametrize("scale", [1, 3, 5])
def test_folding_multiplies_the_gate_count(scale):
    circuit = Circuit.random_circuit(3, 4, seed=1)
    folded = fold_global(circuit, scale)
    assert folded.size == circuit.size * scale


def test_folding_rejects_even_scales():
    circuit = Circuit(2).h(0)
    for bad in (0, 2, 4, -1):
        with pytest.raises(ValueError):
            fold_global(circuit, bad)


def test_folding_increases_noise():
    """The whole point: more gates, more accumulated error."""
    molecule = load_molecule("H2", run_fci=False)
    hamiltonian = fermionic_hamiltonian(molecule)
    ansatz = UCCSDAnsatz(molecule)
    circuit = ansatz.circuit(np.full(ansatz.num_parameters, 0.15))
    ideal = StatevectorEstimator(hamiltonian).energy(circuit)

    estimator = TrajectoryEstimator(
        hamiltonian, DepolarizingNoise(1e-3, 5e-3), trajectories=2_000, seed=0
    )
    errors = [
        abs(estimator.energy(fold_global(circuit, s)) - ideal) for s in (1, 3, 5)
    ]
    assert errors[0] < errors[1] < errors[2]


# ----------------------------------------------------------------------
# extrapolation
# ----------------------------------------------------------------------
def test_linear_extrapolation_is_exact_for_linear_data():
    result = zero_noise_extrapolation(lambda s: 2.0 + 0.5 * s, (1.0, 3.0, 5.0))
    assert result.mitigated == pytest.approx(2.0)
    assert result.residual == pytest.approx(0.0, abs=1e-12)


def test_richardson_is_exact_for_polynomial_data():
    truth = lambda s: -1.0 + 0.3 * s + 0.07 * s * s
    result = zero_noise_extrapolation(truth, (1.0, 3.0, 5.0), model="richardson")
    assert result.mitigated == pytest.approx(-1.0, abs=1e-9)


def test_exponential_model_recovers_an_exponential_decay():
    truth = lambda s: -1.1 + 0.8 * np.exp(-0.4 * s)
    result = zero_noise_extrapolation(
        truth, (1.0, 2.0, 3.0, 5.0, 7.0), model="exponential"
    )
    assert result.mitigated == pytest.approx(-0.3, abs=1e-6)


def test_linear_model_is_wrong_for_strongly_curved_data():
    """An honest limitation: the fit form is an assumption, not a given."""
    truth = lambda s: -1.1 + 0.8 * np.exp(-0.8 * s)
    linear = zero_noise_extrapolation(truth, (1.0, 3.0, 5.0), model="linear")
    exponential = zero_noise_extrapolation(truth, (1.0, 3.0, 5.0), model="exponential")
    target = truth(0.0)
    assert abs(exponential.mitigated - target) < abs(linear.mitigated - target)


def test_result_reports_improvement():
    result = zero_noise_extrapolation(lambda s: 2.0 + 0.5 * s, (1.0, 3.0, 5.0))
    improvement = result.improvement(exact=2.0)
    assert improvement["error_before"] == pytest.approx(0.5)
    assert improvement["error_after"] < 1e-9
    assert result.unmitigated == pytest.approx(2.5)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        zero_noise_extrapolation(lambda s: s, (1.0, 3.0), model="vibes")


def test_extrapolation_removes_most_of_a_real_noise_bias():
    """End to end on a molecule: ZNE must beat doing nothing."""
    molecule = load_molecule("H2")
    hamiltonian = fermionic_hamiltonian(molecule)
    ansatz = UCCSDAnsatz(molecule)
    circuit = ansatz.circuit(np.full(ansatz.num_parameters, 0.1))
    ideal = StatevectorEstimator(hamiltonian).energy(circuit)

    estimator = TrajectoryEstimator(
        hamiltonian, DepolarizingNoise(3e-5, 3e-4), trajectories=4_000, seed=5
    )
    scales = (1.0, 3.0, 5.0)
    measured = {s: estimator.energy(fold_global(circuit, int(s))) for s in scales}
    result = zero_noise_extrapolation(lambda s: measured[s], scales, model="linear")
    improvement = result.improvement(ideal)
    assert improvement["error_after"] < improvement["error_before"]


# ----------------------------------------------------------------------
# readout mitigation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("error", [0.01, 0.05, 0.15])
def test_readout_mitigation_recovers_the_distribution(error):
    mitigator = ReadoutMitigator(3, error, seed=0)
    rng = np.random.default_rng(0)
    truth = rng.random(8)
    truth /= truth.sum()

    observed = mitigator.apply_noise(truth)
    recovered = mitigator.mitigate(observed)

    before = np.abs(observed - truth).sum() / 2
    after = np.abs(recovered - truth).sum() / 2
    assert after < before / 10
    assert after < 1e-9


def test_calibration_matrix_is_stochastic():
    matrix = ReadoutMitigator(3, 0.08).matrix
    assert np.allclose(matrix.sum(axis=0), 1.0)
    assert np.all(matrix >= 0)


def test_mitigated_output_is_a_valid_distribution():
    """Matrix inversion can go slightly negative; the result must not."""
    mitigator = ReadoutMitigator(2, 0.1, seed=0)
    noisy = np.array([0.7, 0.1, 0.1, 0.1])
    noisy = noisy + np.array([0.02, -0.02, 0.01, -0.01])    # pretend shot noise
    recovered = mitigator.mitigate(noisy)
    assert np.all(recovered >= 0)
    assert recovered.sum() == pytest.approx(1.0)


def test_zero_readout_error_is_the_identity():
    mitigator = ReadoutMitigator(2, 0.0)
    assert np.allclose(mitigator.matrix, np.eye(4))


def test_full_calibration_refuses_large_registers():
    with pytest.raises(ValueError):
        ReadoutMitigator(14, 0.01)
