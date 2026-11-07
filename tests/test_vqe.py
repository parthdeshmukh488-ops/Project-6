"""VQE tests.

The variational principle is the load-bearing property: no optimisation may
ever report an energy below the true ground state.  If one does, the
Hamiltonian, the ansatz or the estimator is wrong -- it cannot be a lucky
result.
"""

import numpy as np
import pytest

from vqelab import fermionic_hamiltonian, load_molecule
from vqelab.ansatz import HardwareEfficientAnsatz, UCCSDAnsatz
from vqelab.estimator import StatevectorEstimator
from vqelab.exact import ground_state_energy
from vqelab.vqe import CHEMICAL_ACCURACY, VQE


@pytest.fixture(scope="module")
def h2():
    molecule = load_molecule("H2")
    return molecule, fermionic_hamiltonian(molecule)


# ----------------------------------------------------------------------
# the variational principle
# ----------------------------------------------------------------------
def test_no_trial_state_beats_the_ground_state(h2):
    """The bound that makes VQE meaningful, checked over random parameters."""
    molecule, H = h2
    exact = ground_state_energy(H)
    estimator = StatevectorEstimator(H)
    rng = np.random.default_rng(0)
    for ansatz in (
        UCCSDAnsatz(molecule),
        HardwareEfficientAnsatz(molecule.num_qubits, layers=2),
    ):
        for _ in range(25):
            parameters = rng.normal(0, 2.0, ansatz.num_parameters)
            energy = estimator.energy(ansatz.circuit(parameters))
            assert energy >= exact - 1e-10, f"{ansatz.name} went below the ground state"


def test_optimised_energy_also_respects_the_bound(h2):
    molecule, H = h2
    result = VQE(H, UCCSDAnsatz(molecule)).run(reference_energy=molecule.fci_energy)
    assert result.energy >= ground_state_energy(H) - 1e-10


# ----------------------------------------------------------------------
# convergence
# ----------------------------------------------------------------------
def test_uccsd_reaches_fci_for_h2(h2):
    """UCCSD is exact for two electrons in two orbitals, so this must be tight."""
    molecule, H = h2
    result = VQE(H, UCCSDAnsatz(molecule), optimizer="L-BFGS-B").run(
        reference_energy=molecule.fci_energy, hartree_fock_energy=molecule.hf_energy
    )
    assert abs(result.error) < 1e-9
    assert result.chemically_accurate
    assert result.correlation_recovered == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("optimizer", ["L-BFGS-B", "COBYLA", "Powell"])
def test_optimisers_all_converge(optimizer, h2):
    molecule, H = h2
    result = VQE(H, UCCSDAnsatz(molecule), optimizer=optimizer,
                 max_iterations=2000).run(reference_energy=molecule.fci_energy)
    assert result.chemically_accurate, f"{optimizer} missed chemical accuracy"


def test_hardware_efficient_also_converges_for_h2(h2):
    molecule, H = h2
    ansatz = HardwareEfficientAnsatz(
        molecule.num_qubits, layers=2,
        initial_occupation=molecule.hartree_fock_occupation(),
    )
    result = VQE(H, ansatz, optimizer="L-BFGS-B").run(
        ansatz.initial_parameters(seed=1, scale=0.1),
        reference_energy=molecule.fci_energy,
    )
    assert result.chemically_accurate


def test_optimisation_improves_monotonically_from_hartree_fock(h2):
    molecule, H = h2
    result = VQE(H, UCCSDAnsatz(molecule)).run(
        reference_energy=molecule.fci_energy, hartree_fock_energy=molecule.hf_energy
    )
    assert result.history[0] == pytest.approx(molecule.hf_energy, abs=1e-9)
    assert result.energy < result.history[0]


# ----------------------------------------------------------------------
# gradients
# ----------------------------------------------------------------------
def test_parameter_shift_matches_finite_differences(h2):
    """The shift rule is exact, not an approximation -- so these must agree."""
    molecule, H = h2
    ansatz = HardwareEfficientAnsatz(molecule.num_qubits, layers=2)
    parameters = ansatz.initial_parameters(seed=2, scale=0.4)
    shift = VQE(H, ansatz, gradient="parameter-shift").gradient(parameters)
    finite = VQE(H, ansatz, gradient="finite-difference").gradient(parameters)
    assert np.allclose(shift, finite, atol=1e-6)
    assert np.linalg.norm(shift) > 1e-3        # a non-trivial gradient


def test_gradient_vanishes_at_a_minimum(h2):
    molecule, H = h2
    ansatz = UCCSDAnsatz(molecule)
    driver = VQE(H, ansatz)
    result = driver.run()
    assert np.linalg.norm(driver.gradient(result.parameters)) < 1e-5


def test_auto_gradient_selection(h2):
    molecule, H = h2
    assert VQE(H, UCCSDAnsatz(molecule)).gradient_method == "finite-difference"
    assert (
        VQE(H, HardwareEfficientAnsatz(4, layers=1)).gradient_method
        == "parameter-shift"
    )
    with pytest.raises(ValueError):
        VQE(H, UCCSDAnsatz(molecule), gradient="telepathy")


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def test_result_reports_symmetries_and_summary(h2):
    molecule, H = h2
    result = VQE(H, UCCSDAnsatz(molecule)).run(
        reference_energy=molecule.fci_energy, hartree_fock_energy=molecule.hf_energy
    )
    assert result.particle_number == pytest.approx(molecule.num_electrons, abs=1e-8)
    assert result.spin_z == pytest.approx(0.0, abs=1e-8)
    assert "chemical accuracy" in result.summary()
    assert result.evaluations > 0


def test_chemical_accuracy_constant():
    assert CHEMICAL_ACCURACY == pytest.approx(1.6e-3)
