"""Turning a circuit into an energy.

VQE needs ``<psi|H|psi>`` for a Hamiltonian with hundreds of Pauli terms.
How that number is obtained is the difference between a tidy simulation and
something resembling a real experiment, so both are here.

**Exact.**  Apply the Hamiltonian to the state vector and take the overlap.
One pass over all terms, no sampling noise.  This is what the optimiser uses,
because it isolates the question "can the ansatz and optimiser find the
ground state?" from "can we measure it precisely enough?".

**Sampled.**  What hardware actually does: prepare the state, rotate into a
measurement basis, read out all qubits, repeat.  Each Pauli term's
expectation is a sample mean, with error falling only as ``1/sqrt(shots)``.
This is the dominant practical cost of VQE and the reason measurement
reduction matters.

Measurement grouping
--------------------
Measuring every term separately would need one circuit per term -- hundreds
per energy evaluation, thousands of times over an optimisation.  But Pauli
strings that are *qubit-wise commuting* -- agreeing on every qubit where both
act non-trivially -- can be read from the same measurement, because a single
basis rotation diagonalises all of them at once.

Greedy grouping typically cuts the number of measurement settings by an order
of magnitude, and the reduction is reported rather than assumed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qvolve import Circuit, StateVector

from .pauli import PauliOperator

__all__ = [
    "Estimator",
    "StatevectorEstimator",
    "SampledEstimator",
    "qubitwise_commuting_groups",
]


def _compatible(a: str, b: str) -> bool:
    """Do two Pauli labels agree wherever both act non-trivially?"""
    return all(x == y or x == "I" or y == "I" for x, y in zip(a, b))


def qubitwise_commuting_groups(
    labels: Sequence[str],
) -> List[Tuple[str, List[int]]]:
    """Greedily partition Pauli labels into qubit-wise commuting groups.

    Returns ``(basis, members)`` pairs, where ``basis`` is the label whose
    measurement basis diagonalises every member of the group.
    """
    groups: List[Tuple[List[str], List[int]]] = []
    bases: List[str] = []
    for index, label in enumerate(labels):
        placed = False
        for position, basis in enumerate(bases):
            if _compatible(basis, label):
                merged = "".join(
                    b if b != "I" else l for b, l in zip(basis, label)
                )
                bases[position] = merged
                groups[position][1].append(index)
                placed = True
                break
        if not placed:
            bases.append(label)
            groups.append(([label], [index]))
    return [(bases[i], groups[i][1]) for i in range(len(bases))]


class Estimator(ABC):
    """Maps (circuit, Hamiltonian) to an energy."""

    name = "estimator"

    def __init__(self, hamiltonian: PauliOperator) -> None:
        self.hamiltonian = hamiltonian
        self.evaluations = 0

    @abstractmethod
    def energy(self, circuit: Circuit) -> float:
        ...

    def reset_counter(self) -> None:
        self.evaluations = 0


class StatevectorEstimator(Estimator):
    """Exact expectation value, no sampling noise."""

    name = "statevector"

    def __init__(self, hamiltonian: PauliOperator, fuse: Optional[bool] = None) -> None:
        super().__init__(hamiltonian)
        self.fuse = fuse

    def state(self, circuit: Circuit) -> np.ndarray:
        return circuit.run(fuse=self.fuse).statevector.data

    def energy(self, circuit: Circuit) -> float:
        self.evaluations += 1
        vector = self.state(circuit)
        value = complex(np.vdot(vector, self.hamiltonian.apply(vector)))
        if abs(value.imag) > 1e-8:
            raise ValueError(f"energy came out complex: {value}")
        return float(value.real)


class SampledEstimator(Estimator):
    """Shot-based estimation, grouping qubit-wise commuting terms.

    The constant (identity) term needs no measurement at all -- it is a known
    number -- which matters because for molecules it is large and would
    otherwise dominate the sampling error.
    """

    name = "sampled"

    def __init__(
        self,
        hamiltonian: PauliOperator,
        shots: int = 8192,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(hamiltonian)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

        self.constant = float(np.real(hamiltonian.constant()))
        self.labels: List[str] = []
        self.coefficients: List[float] = []
        identity = "I" * hamiltonian.num_qubits
        for label, coefficient in hamiltonian.label_terms():
            if label == identity:
                continue
            if abs(coefficient.imag) > 1e-9:
                raise ValueError(f"non-real coefficient {coefficient} on {label}")
            self.labels.append(label)
            self.coefficients.append(float(coefficient.real))
        self.groups = qubitwise_commuting_groups(self.labels)

    @property
    def num_settings(self) -> int:
        """Distinct measurement circuits needed per energy evaluation."""
        return len(self.groups)

    @property
    def grouping_reduction(self) -> float:
        if not self.labels:
            return 1.0
        return len(self.labels) / max(len(self.groups), 1)

    def _rotate(self, circuit: Circuit, basis: str) -> Circuit:
        """Append the basis change that maps ``basis`` onto all-Z."""
        rotated = Circuit(circuit.num_qubits, name="measure")
        rotated.compose(circuit)
        for qubit, char in enumerate(basis):
            if char == "X":
                rotated.h(qubit)
            elif char == "Y":
                rotated.rx(np.pi / 2, qubit)
        return rotated

    def energy(self, circuit: Circuit) -> float:
        self.evaluations += 1
        total = self.constant
        shots_per_group = max(1, self.shots // max(len(self.groups), 1))

        for basis, members in self.groups:
            result = self._rotate(circuit, basis).run(
                shots=shots_per_group, seed=int(self.rng.integers(2**31))
            )
            counts = result.counts
            outcomes = np.array(
                [[int(b) for b in bits] for bits in counts], dtype=np.int8
            )
            weights = np.array(list(counts.values()), dtype=float)
            weights /= weights.sum()
            for index in members:
                label = self.labels[index]
                active = [q for q, c in enumerate(label) if c != "I"]
                if not active:
                    continue
                parity = outcomes[:, active].sum(axis=1) & 1
                value = float(np.sum(weights * np.where(parity, -1.0, 1.0)))
                total += self.coefficients[index] * value
        return total
