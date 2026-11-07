"""Ansatz tests: circuit identities, symmetry conservation, expressiveness."""

import numpy as np
import pytest
from qvolve import Circuit
from scipy.linalg import expm

from vqelab import fermionic_hamiltonian, hartree_fock_state, load_molecule
from vqelab.ansatz import (
    HardwareEfficientAnsatz,
    UCCSDAnsatz,
    doubles_excitations,
    excitation_operator,
    pauli_exponential,
    singles_excitations,
)
from vqelab.exact import expectation
from vqelab.hamiltonian import number_operator, spin_z_operator
from vqelab.pauli import PauliOperator


# ----------------------------------------------------------------------
# the Pauli exponential primitive
# ----------------------------------------------------------------------
@pytest.mark.parametrize("label", ["Z", "X", "Y", "ZZ", "XY", "YZX", "XYZI", "IYXZ"])
@pytest.mark.parametrize("theta", [0.3, -1.1, 2.7, np.pi])
def test_pauli_exponential_matches_matrix_exponential(label, theta):
    """The CNOT-ladder construction must equal exp(-i theta P / 2) exactly."""
    n = len(label)
    circuit = Circuit(n)
    pauli_exponential(circuit, label, theta)
    produced = circuit.run(fuse=False).statevector.data

    operator = PauliOperator.from_label(label).to_matrix()
    expected = expm(-0.5j * theta * operator)[:, 0]
    assert np.allclose(produced, expected, atol=1e-12)


def test_pauli_exponential_cost_scales_with_weight():
    """Two CNOTs per extra active qubit -- the ladder there and back."""
    for label, expected in (("ZIII", 0), ("ZZII", 2), ("ZZZI", 4), ("XYZZ", 6)):
        circuit = Circuit(4)
        pauli_exponential(circuit, label, 0.5)
        assert circuit.num_two_qubit_gates == expected


def test_identity_string_and_zero_angle_are_no_ops():
    for label, angle in (("IIII", 0.7), ("XYZI", 0.0)):
        circuit = Circuit(4)
        pauli_exponential(circuit, label, angle)
        assert circuit.size == 0


# ----------------------------------------------------------------------
# excitations
# ----------------------------------------------------------------------
def test_excitation_counts_for_h2():
    """H2/STO-3G: 2 electrons in 4 spin orbitals -> 2 singles, 1 double."""
    assert len(singles_excitations(2, 4)) == 2
    assert len(doubles_excitations(2, 4)) == 1


def test_excitations_conserve_spin():
    for i, a in singles_excitations(4, 12):
        assert i % 2 == a % 2
    for i, j, a, b in doubles_excitations(4, 12):
        assert (i % 2) + (j % 2) == (a % 2) + (b % 2)


@pytest.mark.parametrize("excitation", [(0, 2), (1, 3), (0, 1, 2, 3)])
def test_excitation_generators_are_antihermitian(excitation):
    """Anti-Hermiticity is what makes exp(theta * A) unitary."""
    operator = excitation_operator(4, excitation)
    assert (operator + operator.dagger()).simplify(1e-12).num_terms == 0
    matrix = operator.to_matrix()
    assert np.allclose(matrix, -matrix.conj().T)
    assert np.allclose(expm(0.7 * matrix) @ expm(0.7 * matrix).conj().T, np.eye(16))


def test_excitation_generator_coefficients_are_imaginary():
    for label, coefficient in excitation_operator(4, (0, 1, 2, 3)).label_terms():
        assert abs(coefficient.real) < 1e-12, label


def test_bad_excitation_rejected():
    with pytest.raises(ValueError):
        excitation_operator(4, (0, 1, 2))


# ----------------------------------------------------------------------
# UCCSD
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def h2():
    molecule = load_molecule("H2")
    return molecule, fermionic_hamiltonian(molecule)


def test_uccsd_at_zero_is_hartree_fock(h2):
    """Starting the optimiser at zero therefore starts it at a known energy."""
    molecule, H = h2
    ansatz = UCCSDAnsatz(molecule)
    state = ansatz.circuit(np.zeros(ansatz.num_parameters)).run().statevector.data
    assert np.allclose(np.abs(state), np.abs(hartree_fock_state(molecule)))
    assert expectation(H, state) == pytest.approx(molecule.hf_energy, abs=1e-10)


def test_uccsd_conserves_particle_number_and_spin(h2):
    """The reason to prefer it over a hardware-efficient circuit."""
    molecule, _ = h2
    ansatz = UCCSDAnsatz(molecule)
    rng = np.random.default_rng(0)
    N = number_operator(molecule.num_qubits)
    Sz = spin_z_operator(molecule.num_qubits)
    for _ in range(8):
        parameters = rng.normal(0, 1.5, ansatz.num_parameters)
        state = ansatz.circuit(parameters).run().statevector.data
        assert expectation(N, state) == pytest.approx(molecule.num_electrons, abs=1e-9)
        assert expectation(Sz, state) == pytest.approx(0.0, abs=1e-9)


def test_uccsd_states_stay_normalised(h2):
    molecule, _ = h2
    ansatz = UCCSDAnsatz(molecule)
    rng = np.random.default_rng(1)
    for _ in range(5):
        state = ansatz.circuit(rng.normal(0, 1, ansatz.num_parameters)).run().statevector
        assert state.norm() == pytest.approx(1.0, abs=1e-12)


def test_uccsd_structure(h2):
    molecule, _ = h2
    ansatz = UCCSDAnsatz(molecule)
    assert ansatz.num_parameters == 3
    assert ansatz.num_singles == 2 and ansatz.num_doubles == 1
    assert "H2" in repr(ansatz)
    with pytest.raises(ValueError):
        ansatz.circuit([0.1, 0.2])


def test_singles_only_cannot_beat_hartree_fock(h2):
    """Brillouin's theorem: single excitations alone do not improve on HF.

    It is a real physical statement -- the HF orbitals are chosen precisely
    so that singles have zero gradient -- and a good check that the ansatz
    machinery is faithful.
    """
    molecule, H = h2
    singles = UCCSDAnsatz(molecule, include_doubles=False)
    rng = np.random.default_rng(3)
    best = min(
        expectation(H, singles.circuit(rng.normal(0, 0.3, singles.num_parameters))
                    .run().statevector.data)
        for _ in range(20)
    )
    assert best >= molecule.hf_energy - 1e-9


# ----------------------------------------------------------------------
# hardware-efficient
# ----------------------------------------------------------------------
@pytest.mark.parametrize("layers", [1, 2, 3])
@pytest.mark.parametrize("entangler", ["linear", "circular", "full"])
def test_hardware_efficient_shapes(layers, entangler):
    ansatz = HardwareEfficientAnsatz(4, layers=layers, entangler=entangler)
    assert ansatz.num_parameters == 2 * 4 * (layers + 1)
    circuit = ansatz.circuit(np.zeros(ansatz.num_parameters))
    assert circuit.num_qubits == 4
    assert circuit.run().statevector.norm() == pytest.approx(1.0)


def test_hardware_efficient_leaves_the_physical_sector():
    """Unlike UCCSD it does not conserve electron number -- worth knowing."""
    molecule = load_molecule("H2", run_fci=False)
    ansatz = HardwareEfficientAnsatz(
        molecule.num_qubits, layers=2,
        initial_occupation=molecule.hartree_fock_occupation(),
    )
    N = number_operator(molecule.num_qubits)
    rng = np.random.default_rng(0)
    counts = [
        expectation(N, ansatz.circuit(rng.normal(0, 1.0, ansatz.num_parameters))
                    .run().statevector.data)
        for _ in range(10)
    ]
    assert any(abs(c - molecule.num_electrons) > 0.1 for c in counts)


def test_bad_entangler_rejected():
    with pytest.raises(ValueError):
        HardwareEfficientAnsatz(4, entangler="quantum-glue")


def test_initial_parameters():
    ansatz = HardwareEfficientAnsatz(3, layers=1)
    assert np.allclose(ansatz.initial_parameters(), 0.0)
    spread = ansatz.initial_parameters(seed=0, scale=0.2)
    assert spread.shape == (ansatz.num_parameters,)
    assert np.any(spread != 0.0)
