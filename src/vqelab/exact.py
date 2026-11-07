"""Exact reference energies, for checking everything else.

Nothing in a variational calculation is believable without a number to
compare it against.  Two independent ones are used here: PySCF's FCI energy
(classical quantum chemistry, computed in the orbital basis) and direct
diagonalisation of the qubit Hamiltonian (computed after the
fermion-to-qubit mapping).  They answer slightly different questions -- the
second also tests that the mapping itself is right -- and they must agree to
numerical precision.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .pauli import PauliOperator

__all__ = ["ground_state", "ground_state_energy", "eigenvalues", "expectation"]


def ground_state(
    operator: PauliOperator,
    tolerance: float = 1e-10,
) -> Tuple[float, np.ndarray]:
    """Lowest eigenvalue and eigenvector of a Hermitian Pauli sum.

    Uses Lanczos through a matrix-free operator: the Hamiltonian is never
    built as a matrix, only applied to vectors.  A 14-qubit Hamiltonian with
    a thousand terms would be a 16384 x 16384 dense matrix (4 GiB); applying
    it term by term costs ``O(terms * 2^n)`` per iteration and a few hundred
    megabytes.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    dimension = 1 << operator.num_qubits
    if dimension <= 256:
        matrix = operator.to_matrix()
        values, vectors = np.linalg.eigh(matrix)
        return float(values[0].real), vectors[:, 0]

    linear = LinearOperator(
        (dimension, dimension),
        matvec=operator.apply,
        dtype=complex,
    )
    values, vectors = eigsh(linear, k=1, which="SA", tol=tolerance)
    return float(values[0].real), vectors[:, 0]


def ground_state_energy(operator: PauliOperator, **kwargs) -> float:
    return ground_state(operator, **kwargs)[0]


def eigenvalues(operator: PauliOperator, count: int = 4) -> np.ndarray:
    """The lowest few eigenvalues, for excited states and gap estimates."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    dimension = 1 << operator.num_qubits
    if dimension <= 256:
        return np.linalg.eigvalsh(operator.to_matrix())[:count]
    linear = LinearOperator((dimension, dimension), matvec=operator.apply, dtype=complex)
    return np.sort(eigsh(linear, k=count, which="SA", return_eigenvectors=False))


def expectation(operator: PauliOperator, state: np.ndarray) -> float:
    """``<psi|H|psi>`` for a normalised state, returned as a real number."""
    state = np.asarray(state, dtype=complex).ravel()
    value = complex(np.vdot(state, operator.apply(state)))
    if abs(value.imag) > 1e-8:
        raise ValueError(f"expectation value is not real: {value}")
    return float(value.real)
