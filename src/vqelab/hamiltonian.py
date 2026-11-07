"""From electrons to qubits: the Jordan-Wigner transformation.

The problem
-----------
Electrons are fermions: swapping two of them flips the sign of the
wavefunction.  Creation and annihilation operators encode that through
*anti*-commutation,

    {a_i, a_j^dag} = delta_ij,     {a_i, a_j} = 0

Qubits do not behave like that at all.  Operators on different qubits simply
*commute*, so writing ``a_j`` as something local on qubit ``j`` would silently
lose the minus signs that make electrons electrons -- and with them, the Pauli
exclusion principle.

The Jordan-Wigner fix
---------------------
Attach a string of ``Z`` operators running back to the start of the register:

    a_j = (X_j + i Y_j)/2  (x)  Z_0 Z_1 ... Z_{j-1}

Those ``Z``s count the parity of every orbital below ``j``.  When two such
operators are swapped, the strings overlap on an odd number of qubits and
produce exactly the minus sign fermions demand.  The anticommutation
relations come out right, and the test suite checks all of them rather than
taking it on faith.

The price is locality: an operator on orbital ``j`` touches ``j+1`` qubits, so
Hamiltonian terms can act on the whole register.  Parity and Bravyi-Kitaev
mappings trade that off differently; Jordan-Wigner is chosen here because it
is the one whose correctness you can see.

The Hamiltonian
---------------
    H = E_nuc + sum_pq h_pq a_p^dag a_q
              + 1/2 sum_pqrs (pq|rs) a_{p,sigma}^dag a_{r,tau}^dag a_{s,tau} a_{q,sigma}

in chemist notation, matching what PySCF hands over.  Spin is carried by the
interleaving ``2p + sigma``, and the spin deltas are implicit: an integral
only contributes when the two creation/annihilation pairs keep their spins.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .integrals import MolecularData
from .pauli import PauliOperator, qubit_bit

__all__ = [
    "jordan_wigner_annihilate",
    "jordan_wigner_create",
    "fermionic_hamiltonian",
    "number_operator",
    "spin_z_operator",
    "hartree_fock_state",
]


def _string_below(num_qubits: int, orbital: int) -> int:
    """Mask of the ``Z`` string on every qubit before ``orbital``."""
    mask = 0
    for lower in range(orbital):
        mask |= qubit_bit(num_qubits, lower)
    return mask


def jordan_wigner_annihilate(num_qubits: int, orbital: int) -> PauliOperator:
    """``a_j`` as a Pauli sum.

    Writing ``Y = iXZ``, the combination ``(X + iY)/2`` becomes
    ``(X - XZ)/2``, so the operator is two Pauli strings differing only by a
    ``Z`` on the target qubit -- the ``Z`` string below it is shared.
    """
    bit = qubit_bit(num_qubits, orbital)
    string = _string_below(num_qubits, orbital)
    return PauliOperator(
        num_qubits,
        {(bit, string): 0.5 + 0j, (bit, string | bit): -0.5 + 0j},
    )


def jordan_wigner_create(num_qubits: int, orbital: int) -> PauliOperator:
    """``a_j^dag`` as a Pauli sum: the same two strings, added instead."""
    bit = qubit_bit(num_qubits, orbital)
    string = _string_below(num_qubits, orbital)
    return PauliOperator(
        num_qubits,
        {(bit, string): 0.5 + 0j, (bit, string | bit): 0.5 + 0j},
    )


def _operator_cache(num_qubits: int) -> Tuple[List[PauliOperator], List[PauliOperator]]:
    create = [jordan_wigner_create(num_qubits, p) for p in range(num_qubits)]
    annihilate = [jordan_wigner_annihilate(num_qubits, p) for p in range(num_qubits)]
    return create, annihilate


def fermionic_hamiltonian(
    molecule: MolecularData,
    tolerance: float = 1e-12,
) -> PauliOperator:
    """Build the qubit Hamiltonian of a molecule under Jordan-Wigner.

    Integrals smaller than ``tolerance`` are skipped.  That is not an
    approximation so much as an acknowledgement that most of the
    ``nmo^4`` two-electron integrals vanish by symmetry, and multiplying out
    Pauli strings for them would dominate the runtime.
    """
    nmo = molecule.num_spatial_orbitals
    n = molecule.num_spin_orbitals
    create, annihilate = _operator_cache(n)

    terms: Dict[Tuple[int, int], complex] = {}

    def accumulate(operator: PauliOperator, scale: complex) -> None:
        for key, value in operator.terms.items():
            total = terms.get(key, 0j) + scale * value
            if abs(total) < 1e-16:
                terms.pop(key, None)
            else:
                terms[key] = total

    # Constant: nuclear repulsion (plus any frozen-core energy).
    terms[(0, 0)] = complex(molecule.nuclear_repulsion)

    # One-body: h_pq a_p^dag a_q, diagonal in spin.
    h1 = molecule.one_body
    for p in range(nmo):
        for q in range(nmo):
            value = h1[p, q]
            if abs(value) < tolerance:
                continue
            for spin in (0, 1):
                accumulate(create[2 * p + spin] * annihilate[2 * q + spin], value)

    # Two-body: 1/2 (pq|rs) a_{p,s}^dag a_{r,t}^dag a_{s,t} a_{q,s}
    eri = molecule.two_body
    for p in range(nmo):
        for q in range(nmo):
            for r in range(nmo):
                for s in range(nmo):
                    value = eri[p, q, r, s]
                    if abs(value) < tolerance:
                        continue
                    half = 0.5 * value
                    for sigma in (0, 1):
                        for tau in (0, 1):
                            product = (
                                create[2 * p + sigma]
                                * create[2 * r + tau]
                                * annihilate[2 * s + tau]
                                * annihilate[2 * q + sigma]
                            )
                            accumulate(product, half)

    return PauliOperator(n, terms).simplify(1e-12)


def number_operator(num_qubits: int) -> PauliOperator:
    """Total electron number ``sum_p a_p^dag a_p``.

    Under Jordan-Wigner this is ``sum_p (I - Z_p)/2``: occupation shows up as
    the ``Z`` expectation of each qubit, which is why particle number is so
    cheap to check.
    """
    terms: Dict[Tuple[int, int], complex] = {(0, 0): 0.5 * num_qubits + 0j}
    for p in range(num_qubits):
        terms[(0, qubit_bit(num_qubits, p))] = -0.5 + 0j
    return PauliOperator(num_qubits, terms)


def spin_z_operator(num_qubits: int) -> PauliOperator:
    """Total ``S_z``: ``+1/2`` for each spin-up electron, ``-1/2`` for each down."""
    if num_qubits % 2:
        raise ValueError("spin orbitals come in pairs")
    terms: Dict[Tuple[int, int], complex] = {}
    for p in range(num_qubits):
        sign = 0.5 if p % 2 == 0 else -0.5
        key = (0, qubit_bit(num_qubits, p))
        terms[key] = terms.get(key, 0j) - 0.5 * sign
        terms[(0, 0)] = terms.get((0, 0), 0j) + 0.5 * sign
    return PauliOperator(num_qubits, terms).simplify()


def hartree_fock_state(molecule: MolecularData) -> np.ndarray:
    """The Hartree-Fock reference as a state vector.

    A single computational basis state: occupied spin orbitals are ``|1>``.
    It is the starting point for every ansatz here, and already captures
    ~99% of the total energy -- VQE is only ever chasing the last fraction
    of a percent, which is precisely what makes chemical accuracy hard.
    """
    n = molecule.num_qubits
    index = 0
    for orbital in molecule.hartree_fock_occupation():
        index |= qubit_bit(n, orbital)
    vector = np.zeros(1 << n, dtype=complex)
    vector[index] = 1.0
    return vector
