"""Noise-model tests: trajectories must reproduce the density-matrix answer."""

import numpy as np
import pytest
from qvolve import Circuit

from vqelab import fermionic_hamiltonian, load_molecule
from vqelab.ansatz import UCCSDAnsatz
from vqelab.estimator import StatevectorEstimator
from vqelab.noise import DepolarizingNoise, TrajectoryEstimator, noisy_circuit
from vqelab.pauli import PauliOperator


def test_noise_model_validation():
    DepolarizingNoise(0.0, 0.0)
    for bad in ({"one_qubit": -0.1}, {"two_qubit": 1.5}, {"readout": 2.0}):
        with pytest.raises(ValueError):
            DepolarizingNoise(**bad)


def test_scaling_multiplies_rates_and_clamps():
    model = DepolarizingNoise(1e-3, 1e-2)
    scaled = model.scaled(3.0)
    assert scaled.one_qubit == pytest.approx(3e-3)
    assert scaled.two_qubit == pytest.approx(3e-2)
    assert DepolarizingNoise(0.5, 0.5).scaled(10).two_qubit == 1.0
    assert DepolarizingNoise(0.0, 0.0).is_noiseless


def test_noiseless_model_inserts_nothing():
    circuit = Circuit(3).h(0).cx(0, 1).rz(0.3, 2)
    noisy = noisy_circuit(circuit, DepolarizingNoise(0.0, 0.0), np.random.default_rng(0))
    assert noisy.size == circuit.size


def test_noise_insertion_rate_matches_the_model():
    """Over many samples the number of inserted Paulis must track the rates."""
    circuit = Circuit(2)
    for _ in range(50):
        circuit.cx(0, 1)
    rate = 0.2
    rng = np.random.default_rng(0)
    extra = [
        noisy_circuit(circuit, DepolarizingNoise(0.0, rate), rng).size - circuit.size
        for _ in range(200)
    ]
    # Each faulty gate adds one or two Paulis (a uniform non-identity
    # two-qubit Pauli has one or two non-identity factors), so the mean sits
    # between 1x and 2x the expected number of faults.
    faults = 50 * rate
    assert faults <= np.mean(extra) <= 2 * faults


def test_trajectory_average_matches_the_density_matrix():
    """The key claim: sampling Pauli errors reproduces the channel exactly.

    A single depolarising channel on one qubit is small enough to compute
    both ways, so the trajectory method can be checked rather than trusted.
    """
    p = 0.3
    observable = PauliOperator.from_label("Z")
    circuit = Circuit(1).h(0).ry(0.7, 0)        # some non-trivial state
    pure = circuit.run().statevector.data
    ideal = float(np.real(np.vdot(pure, observable.apply(pure))))

    # Exact depolarising channel on the final state: <Z> -> (1 - 4p/3) <Z>.
    # Noise follows each of the two gates independently.
    expected = ideal * (1 - 4 * p / 3) ** 2

    estimator = TrajectoryEstimator(
        observable, DepolarizingNoise(p, 0.0), trajectories=40_000, seed=0
    )
    measured = estimator.energy(circuit)
    assert abs(measured - expected) < 5 * estimator.last_stderr + 1e-3


def test_trajectory_error_falls_as_one_over_sqrt_n():
    observable = PauliOperator.from_label("ZZ")
    circuit = Circuit(2).h(0).cx(0, 1)
    model = DepolarizingNoise(0.02, 0.05)
    errors = []
    for trajectories in (250, 4_000):
        estimator = TrajectoryEstimator(observable, model, trajectories, seed=3)
        estimator.energy(circuit)
        errors.append(estimator.last_stderr)
    assert errors[1] < errors[0] / 2


def test_noise_biases_the_energy_upward():
    """Noise is a bias, not just scatter -- which is why mitigation exists."""
    molecule = load_molecule("H2", run_fci=False)
    hamiltonian = fermionic_hamiltonian(molecule)
    ansatz = UCCSDAnsatz(molecule)
    circuit = ansatz.circuit(np.full(ansatz.num_parameters, 0.1))
    ideal = StatevectorEstimator(hamiltonian).energy(circuit)

    estimator = TrajectoryEstimator(
        hamiltonian, DepolarizingNoise(1e-3, 1e-2), trajectories=2_000, seed=0
    )
    noisy = estimator.energy(circuit)
    assert noisy > ideal + 5 * estimator.last_stderr


def test_noiseless_trajectory_estimator_is_exact():
    molecule = load_molecule("H2", run_fci=False)
    hamiltonian = fermionic_hamiltonian(molecule)
    ansatz = UCCSDAnsatz(molecule)
    circuit = ansatz.circuit(np.zeros(ansatz.num_parameters))
    estimator = TrajectoryEstimator(hamiltonian, DepolarizingNoise(0.0, 0.0), 5)
    assert estimator.energy(circuit) == pytest.approx(molecule.hf_energy, abs=1e-10)
    assert estimator.last_stderr == 0.0
