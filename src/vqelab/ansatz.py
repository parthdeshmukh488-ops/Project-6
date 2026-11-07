"""Ansatz circuits: the trial states VQE searches over.

VQE minimises ``<psi(theta)|H|psi(theta)>`` over a family of states a quantum
computer can actually prepare.  The family is everything: too small and the
true ground state is out of reach no matter how good the optimiser; too large
and the landscape becomes unoptimisable.

Two families are implemented, and they sit at opposite ends of that trade.

**Hardware-efficient.**  Alternating layers of single-qubit rotations and
nearest-neighbour entanglers.  Shallow, native to the hardware, and makes no
reference to chemistry at all -- which is the problem: it wanders outside the
physical sector (wrong electron number, wrong spin) and its landscape suffers
from barren plateaus as the register grows.

**UCCSD.**  Unitary coupled cluster with singles and doubles, the quantum
version of the classical chemistry workhorse.  Built from fermionic
excitations, so it conserves electron number and spin *by construction* and
starts from Hartree-Fock.  Far deeper, but physically motivated, and its
parameters mean something.

Exponentiating a Pauli string
-----------------------------
Both ansaetze are built from one primitive: a circuit for
``exp(-i theta P / 2)`` where ``P`` is a Pauli string.  The construction is
the standard one and worth understanding, because it is three ideas stacked:

1. rotate each qubit so its Pauli becomes ``Z`` (``H`` for ``X``,
   ``RX(pi/2)`` for ``Y``);
2. a CNOT ladder accumulates the *parity* of all involved qubits onto the
   last one, because ``exp(-i theta Z...Z / 2)`` only depends on that parity;
3. a single ``RZ(theta)`` applies the phase, then everything unwinds.

So an operator on ``k`` qubits costs ``2(k-1)`` CNOTs and one rotation.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qvolve import Circuit

from .hamiltonian import jordan_wigner_annihilate, jordan_wigner_create
from .integrals import MolecularData
from .pauli import PauliOperator

__all__ = [
    "pauli_exponential",
    "Ansatz",
    "HardwareEfficientAnsatz",
    "UCCSDAnsatz",
    "singles_excitations",
    "doubles_excitations",
    "excitation_operator",
]


def pauli_exponential(circuit: Circuit, label: str, angle: float) -> Circuit:
    """Append ``exp(-i * angle * P / 2)`` for the Pauli string ``label``.

    The identity string is a global phase and is skipped.
    """
    active = [q for q, char in enumerate(label) if char != "I"]
    if not active or angle == 0.0:
        return circuit

    for q in active:
        if label[q] == "X":
            circuit.h(q)
        elif label[q] == "Y":
            circuit.rx(np.pi / 2, q)

    for a, b in zip(active, active[1:]):
        circuit.cx(a, b)
    circuit.rz(angle, active[-1])
    for a, b in reversed(list(zip(active, active[1:]))):
        circuit.cx(a, b)

    for q in active:
        if label[q] == "X":
            circuit.h(q)
        elif label[q] == "Y":
            circuit.rx(-np.pi / 2, q)
    return circuit


class Ansatz(ABC):
    """A parameterised family of circuits."""

    name = "ansatz"

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        ...

    @abstractmethod
    def circuit(self, parameters: Sequence[float]) -> Circuit:
        ...

    def initial_parameters(self, seed: Optional[int] = None, scale: float = 0.0):
        """Starting point for the optimiser.

        Zero is the natural choice for UCCSD, where it reproduces exactly the
        Hartree-Fock state; a small random spread helps hardware-efficient
        circuits escape the symmetric point where all gradients vanish.
        """
        if scale == 0.0:
            return np.zeros(self.num_parameters)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, scale, self.num_parameters)

    def depth(self, parameters: Optional[Sequence[float]] = None) -> int:
        params = (
            np.full(self.num_parameters, 0.1) if parameters is None else parameters
        )
        return self.circuit(params).depth()

    def two_qubit_gate_count(self, parameters: Optional[Sequence[float]] = None) -> int:
        params = (
            np.full(self.num_parameters, 0.1) if parameters is None else parameters
        )
        return self.circuit(params).num_two_qubit_gates


# ======================================================================
# hardware-efficient
# ======================================================================
class HardwareEfficientAnsatz(Ansatz):
    """``RY``-``RZ`` rotations alternating with a linear CNOT chain.

    Deliberately ignorant of chemistry.  It is cheap and hardware-native, but
    nothing stops it leaving the correct particle-number sector -- which is
    worth measuring rather than assuming, so ``vqe.py`` reports ``<N>`` on the
    optimised state.
    """

    name = "hardware-efficient"

    def __init__(
        self,
        num_qubits: int,
        layers: int = 2,
        initial_occupation: Optional[Sequence[int]] = None,
        entangler: str = "linear",
    ) -> None:
        super().__init__(num_qubits)
        self.layers = layers
        self.initial_occupation = list(initial_occupation or [])
        if entangler not in ("linear", "circular", "full"):
            raise ValueError(f"unknown entangler {entangler!r}")
        self.entangler = entangler

    @property
    def num_parameters(self) -> int:
        return 2 * self.num_qubits * (self.layers + 1)

    def _entangle(self, circuit: Circuit) -> None:
        n = self.num_qubits
        if self.entangler == "full":
            for a, b in itertools.combinations(range(n), 2):
                circuit.cx(a, b)
            return
        for q in range(n - 1):
            circuit.cx(q, q + 1)
        if self.entangler == "circular" and n > 2:
            circuit.cx(n - 1, 0)

    def circuit(self, parameters: Sequence[float]) -> Circuit:
        parameters = np.asarray(parameters, dtype=float)
        if parameters.size != self.num_parameters:
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {parameters.size}"
            )
        qc = Circuit(self.num_qubits, name="hea")
        for q in self.initial_occupation:
            qc.x(q)
        index = 0
        for layer in range(self.layers + 1):
            for q in range(self.num_qubits):
                qc.ry(float(parameters[index]), q)
                index += 1
                qc.rz(float(parameters[index]), q)
                index += 1
            if layer < self.layers:
                self._entangle(qc)
        return qc


# ======================================================================
# UCCSD
# ======================================================================
def singles_excitations(num_electrons: int, num_spin_orbitals: int) -> List[Tuple]:
    """Spin-conserving single excitations ``occupied -> virtual``."""
    out = []
    for i in range(num_electrons):
        for a in range(num_electrons, num_spin_orbitals):
            if (i % 2) == (a % 2):              # same spin
                out.append((i, a))
    return out


def doubles_excitations(num_electrons: int, num_spin_orbitals: int) -> List[Tuple]:
    """Spin-conserving double excitations ``(i,j) -> (a,b)``."""
    out = []
    occupied = range(num_electrons)
    virtual = range(num_electrons, num_spin_orbitals)
    for i, j in itertools.combinations(occupied, 2):
        for a, b in itertools.combinations(virtual, 2):
            if (i % 2) + (j % 2) == (a % 2) + (b % 2):   # total spin preserved
                out.append((i, j, a, b))
    return out


def excitation_operator(num_qubits: int, excitation: Tuple[int, ...]) -> PauliOperator:
    """The anti-Hermitian generator ``T - T^dag`` for one excitation.

    Anti-Hermiticity is what makes ``exp(theta * (T - T^dag))`` unitary, and
    it is asserted in the tests rather than assumed.
    """
    create = lambda p: jordan_wigner_create(num_qubits, p)
    annihilate = lambda p: jordan_wigner_annihilate(num_qubits, p)

    if len(excitation) == 2:
        i, a = excitation
        term = create(a) * annihilate(i)
    elif len(excitation) == 4:
        i, j, a, b = excitation
        term = create(a) * create(b) * annihilate(j) * annihilate(i)
    else:
        raise ValueError(f"excitation must have 2 or 4 indices, got {excitation}")
    return (term - term.dagger()).simplify()


class UCCSDAnsatz(Ansatz):
    """Unitary coupled cluster, singles and doubles, first-order Trotterised.

    ``|psi(theta)> = prod_k exp(theta_k (T_k - T_k^dag)) |HF>``

    One parameter per excitation.  At ``theta = 0`` the circuit is the
    identity and the state is exactly Hartree-Fock, which is why the
    optimiser can safely start from zero -- it begins at a known, physically
    sensible energy and only improves.

    Trotterising the exponential is an approximation, but a harmless one
    here: the product of individual exponentials is still a valid unitary
    parameterised by ``theta``, so the variational principle holds regardless.
    It is a *different* ansatz, not a worse-approximated one.
    """

    name = "uccsd"

    def __init__(
        self,
        molecule: MolecularData,
        include_singles: bool = True,
        include_doubles: bool = True,
        trotter_steps: int = 1,
    ) -> None:
        super().__init__(molecule.num_qubits)
        self.molecule = molecule
        self.trotter_steps = trotter_steps
        n, ne = molecule.num_spin_orbitals, molecule.num_electrons

        self.excitations: List[Tuple[int, ...]] = []
        if include_singles:
            self.excitations += singles_excitations(ne, n)
        if include_doubles:
            self.excitations += doubles_excitations(ne, n)

        # Cache each generator's Pauli decomposition once.  Every term of an
        # anti-Hermitian Pauli sum has a purely imaginary coefficient, so the
        # real part below is the rotation angle.
        self._generators: List[List[Tuple[str, float]]] = []
        for excitation in self.excitations:
            operator = excitation_operator(self.num_qubits, excitation)
            decomposition = []
            for label, coefficient in operator.label_terms():
                if abs(coefficient) < 1e-12:
                    continue
                if abs(coefficient.real) > 1e-9:
                    raise ValueError(
                        "anti-Hermitian generator produced a real coefficient "
                        f"({coefficient}) for {label}"
                    )
                decomposition.append((label, float(coefficient.imag)))
            self._generators.append(decomposition)

    @property
    def num_parameters(self) -> int:
        return len(self.excitations)

    @property
    def num_singles(self) -> int:
        return sum(1 for e in self.excitations if len(e) == 2)

    @property
    def num_doubles(self) -> int:
        return sum(1 for e in self.excitations if len(e) == 4)

    def circuit(self, parameters: Sequence[float]) -> Circuit:
        parameters = np.asarray(parameters, dtype=float)
        if parameters.size != self.num_parameters:
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {parameters.size}"
            )
        qc = Circuit(self.num_qubits, name="uccsd")
        for q in self.molecule.hartree_fock_occupation():
            qc.x(q)

        step = 1.0 / self.trotter_steps
        for _ in range(self.trotter_steps):
            for theta, decomposition in zip(parameters, self._generators):
                if theta == 0.0:
                    continue
                for label, imaginary in decomposition:
                    # exp(theta * i * c * P) = exp(-i * (-2 theta c) * P / 2)
                    pauli_exponential(qc, label, -2.0 * step * float(theta) * imaginary)
        return qc

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"UCCSDAnsatz({self.molecule.name}, {self.num_qubits} qubits, "
            f"{self.num_singles} singles + {self.num_doubles} doubles)"
        )
