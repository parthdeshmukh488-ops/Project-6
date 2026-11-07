"""Estimator tests: exact energies, measurement grouping, shot statistics."""

import numpy as np
import pytest
from qvolve import Circuit

from vqelab import fermionic_hamiltonian, load_molecule
from vqelab.ansatz import UCCSDAnsatz
from vqelab.estimator import (
    SampledEstimator,
    StatevectorEstimator,
    qubitwise_commuting_groups,
)
from vqelab.pauli import PauliOperator


@pytest.fixture(scope="module")
def h2():
    molecule = load_molecule("H2")
    return molecule, fermionic_hamiltonian(molecule)


# ----------------------------------------------------------------------
# grouping
# ----------------------------------------------------------------------
def test_grouping_merges_compatible_strings():
    groups = qubitwise_commuting_groups(["ZII", "IZI", "IIZ", "ZZZ"])
    assert len(groups) == 1                 # all diagonal in the Z basis
    assert groups[0][0] == "ZZZ"


def test_grouping_separates_incompatible_strings():
    groups = qubitwise_commuting_groups(["XII", "YII", "ZII"])
    assert len(groups) == 3                 # three different bases on qubit 0


def test_every_label_lands_in_exactly_one_group():
    labels = ["XYZ", "XYI", "ZZZ", "IIX", "YYY", "XIZ"]
    groups = qubitwise_commuting_groups(labels)
    members = [i for _, group in groups for i in group]
    assert sorted(members) == list(range(len(labels)))


def test_group_basis_is_compatible_with_all_its_members():
    """The basis must agree with every member wherever that member acts."""
    labels = ["XYZ", "XYI", "ZZZ", "IIX", "YYY", "XIZ", "XZI"]
    for basis, members in qubitwise_commuting_groups(labels):
        for index in members:
            for b, l in zip(basis, labels[index]):
                assert l == "I" or b == l


def test_grouping_reduces_measurements_for_a_molecule(h2):
    _, H = h2
    estimator = SampledEstimator(H, shots=100)
    assert estimator.num_settings < len(estimator.labels)
    assert estimator.grouping_reduction > 1.5


# ----------------------------------------------------------------------
# exact estimation
# ----------------------------------------------------------------------
def test_statevector_estimator_matches_direct_expectation(h2):
    molecule, H = h2
    ansatz = UCCSDAnsatz(molecule)
    estimator = StatevectorEstimator(H)
    rng = np.random.default_rng(0)
    for _ in range(5):
        parameters = rng.normal(0, 0.5, ansatz.num_parameters)
        circuit = ansatz.circuit(parameters)
        state = circuit.run().statevector.data
        direct = float(np.real(np.vdot(state, H.apply(state))))
        assert estimator.energy(circuit) == pytest.approx(direct, abs=1e-12)


def test_evaluation_counter(h2):
    _, H = h2
    estimator = StatevectorEstimator(H)
    circuit = Circuit(H.num_qubits).h(0)
    for _ in range(3):
        estimator.energy(circuit)
    assert estimator.evaluations == 3
    estimator.reset_counter()
    assert estimator.evaluations == 0


# ----------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------
def test_sampled_energy_converges_to_the_exact_value(h2):
    """Shot noise must shrink as 1/sqrt(shots) and centre on the truth."""
    molecule, H = h2
    ansatz = UCCSDAnsatz(molecule)
    circuit = ansatz.circuit(np.zeros(ansatz.num_parameters))
    exact = StatevectorEstimator(H).energy(circuit)

    spreads = []
    for shots in (2_000, 32_000):
        samples = [
            SampledEstimator(H, shots=shots, seed=s).energy(circuit) for s in range(8)
        ]
        spreads.append(float(np.std(samples)))
        assert abs(float(np.mean(samples)) - exact) < 0.02
    assert spreads[1] < spreads[0]           # more shots, less noise


def test_identity_term_costs_no_measurements(h2):
    """The constant is known exactly, so it must not be sampled."""
    _, H = h2
    estimator = SampledEstimator(H, shots=100)
    identity = "I" * H.num_qubits
    assert identity not in estimator.labels
    assert estimator.constant == pytest.approx(float(np.real(H.constant())))


def test_sampled_estimator_rejects_complex_coefficients():
    operator = PauliOperator.from_label("XY", 1j)
    with pytest.raises(ValueError):
        SampledEstimator(operator, shots=10)
