"""Jordan-Wigner and molecular Hamiltonian tests.

The headline check is that the qubit Hamiltonian reproduces PySCF's FCI
energy.  That is a strong test: it exercises the integral handling, the
spin-orbital expansion, the operator ordering conventions and the
fermion-to-qubit mapping all at once, against a completely independent
implementation of the same physics.
"""

import itertools

import numpy as np
import pytest

from vqelab import fermionic_hamiltonian, hartree_fock_state, load_molecule
from vqelab.exact import expectation, ground_state_energy
from vqelab.hamiltonian import (
    jordan_wigner_annihilate,
    jordan_wigner_create,
    number_operator,
    spin_z_operator,
)
from vqelab.pauli import PauliOperator


# ----------------------------------------------------------------------
# the mapping itself
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_anticommutation_relations(n):
    """{a_i, a_j^dag} = delta_ij and {a_i, a_j} = 0.

    These *define* fermions.  Jordan-Wigner's Z strings exist solely to make
    them come out right, so this is the test that the mapping is a mapping.
    """
    dimension = 1 << n
    identity = np.eye(dimension)
    for i, j in itertools.product(range(n), repeat=2):
        a_i = jordan_wigner_annihilate(n, i).to_matrix()
        a_j = jordan_wigner_annihilate(n, j).to_matrix()
        ad_j = jordan_wigner_create(n, j).to_matrix()

        anticommutator = a_i @ ad_j + ad_j @ a_i
        expected = identity if i == j else np.zeros((dimension, dimension))
        assert np.allclose(anticommutator, expected), f"{{a_{i}, a_{j}^dag}}"

        assert np.allclose(a_i @ a_j + a_j @ a_i, 0), f"{{a_{i}, a_{j}}}"


@pytest.mark.parametrize("n", [1, 3, 4])
def test_creation_is_the_adjoint_of_annihilation(n):
    for p in range(n):
        a = jordan_wigner_annihilate(n, p).to_matrix()
        ad = jordan_wigner_create(n, p).to_matrix()
        assert np.allclose(ad, a.conj().T)


def test_annihilation_empties_an_occupied_orbital():
    """On one qubit, a|1> = |0> and a|0> = 0."""
    a = jordan_wigner_annihilate(1, 0).to_matrix()
    assert np.allclose(a @ np.array([0, 1]), np.array([1, 0]))
    assert np.allclose(a @ np.array([1, 0]), 0)


def test_number_operator_counts_occupation():
    n = 4
    N = number_operator(n)
    for occupied in ([], [0], [1, 3], [0, 1, 2, 3]):
        index = sum(1 << (n - 1 - q) for q in occupied)
        state = np.zeros(1 << n)
        state[index] = 1.0
        assert expectation(N, state) == pytest.approx(len(occupied))


def test_spin_operator_counts_polarisation():
    n = 4           # spin orbitals 0,2 are up; 1,3 are down
    Sz = spin_z_operator(n)
    for occupied, expected in ([([0], 0.5), ([1], -0.5), ([0, 1], 0.0), ([0, 2], 1.0)]):
        index = sum(1 << (n - 1 - q) for q in occupied)
        state = np.zeros(1 << n)
        state[index] = 1.0
        assert expectation(Sz, state) == pytest.approx(expected)


def test_number_operator_commutes_with_the_hamiltonian():
    """Electron number is conserved, so the Hamiltonian must not change it."""
    molecule = load_molecule("H2", run_fci=False)
    H = fermionic_hamiltonian(molecule)
    N = number_operator(molecule.num_qubits)
    assert (H * N - N * H).simplify(1e-10).num_terms == 0


# ----------------------------------------------------------------------
# molecular energies, against PySCF
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def h2():
    molecule = load_molecule("H2")
    return molecule, fermionic_hamiltonian(molecule)


def test_hamiltonian_is_hermitian(h2):
    _, H = h2
    assert H.is_hermitian()


def test_hartree_fock_energy_matches_pyscf(h2):
    molecule, H = h2
    energy = expectation(H, hartree_fock_state(molecule))
    assert energy == pytest.approx(molecule.hf_energy, abs=1e-10)


def test_ground_state_matches_fci(h2):
    """The whole mapping, validated end to end against classical chemistry."""
    molecule, H = h2
    assert ground_state_energy(H) == pytest.approx(molecule.fci_energy, abs=1e-9)


def test_hartree_fock_state_has_the_right_symmetries(h2):
    molecule, H = h2
    state = hartree_fock_state(molecule)
    assert expectation(number_operator(molecule.num_qubits), state) == pytest.approx(
        molecule.num_electrons
    )
    assert expectation(spin_z_operator(molecule.num_qubits), state) == pytest.approx(0.0)
    assert np.isclose(np.linalg.norm(state), 1.0)


@pytest.mark.parametrize("bond_length", [0.5, 0.735, 1.2, 2.0])
def test_h2_dissociation_matches_fci(bond_length):
    """Including the stretched geometries, where Hartree-Fock breaks down."""
    molecule = load_molecule("H2", bond_length=bond_length)
    H = fermionic_hamiltonian(molecule)
    assert ground_state_energy(H) == pytest.approx(molecule.fci_energy, abs=1e-8)


@pytest.mark.slow
def test_lih_matches_fci():
    molecule = load_molecule("LiH")
    H = fermionic_hamiltonian(molecule)
    assert molecule.num_qubits == 12
    assert ground_state_energy(H) == pytest.approx(molecule.fci_energy, abs=1e-7)


@pytest.mark.slow
def test_frozen_core_stays_within_chemical_accuracy():
    """Freezing the lithium 1s saves two qubits and costs almost nothing."""
    from vqelab.vqe import CHEMICAL_ACCURACY

    full = load_molecule("LiH")
    frozen = load_molecule("LiH", frozen_core=1)
    assert frozen.num_qubits == full.num_qubits - 2
    energy = ground_state_energy(fermionic_hamiltonian(frozen))
    assert abs(energy - full.fci_energy) < CHEMICAL_ACCURACY


def test_term_count_is_reasonable(h2):
    _, H = h2
    assert H.num_terms == 15          # the textbook count for H2/STO-3G
    assert H.max_weight() == 4


def test_unknown_molecule_rejected():
    with pytest.raises(ValueError):
        load_molecule("unobtainium")
    with pytest.raises(ValueError):
        load_molecule("H2O", bond_length=1.0)      # not a diatomic
