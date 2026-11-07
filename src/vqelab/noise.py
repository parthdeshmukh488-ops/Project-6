"""Noise models, simulated by quantum trajectories.

``qvolve`` stores a pure state, not a density matrix.  That looks like a
limitation for noise, and it is not: a Pauli channel can be sampled instead
of averaged.  For a depolarising channel

    rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)

you can draw one of ``{I, X, Y, Z}`` with the matching probability, apply it
to the pure state, and average observables over many such *trajectories*.
The average converges to exactly the density-matrix answer, and each
trajectory costs ``2^n`` amplitudes rather than ``4^n``.

The cost is statistical: trajectory noise adds its own error bar on top of
shot noise, falling as ``1/sqrt(trajectories)``.  That is a real trade and is
reported alongside every noisy energy here rather than hidden.

Why this matters for VQE
------------------------
A noisy expectation value is *biased*, not merely noisy.  Errors push the
state towards the maximally mixed one, which drags the measured energy
towards the Hamiltonian's trace -- usually upward, and by far more than
chemical accuracy at realistic gate error rates.  Averaging harder does not
help.  Removing that bias is what :mod:`vqelab.mitigation` is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from qvolve import Circuit
from qvolve import gates as G

from .estimator import Estimator
from .pauli import PauliOperator

__all__ = ["DepolarizingNoise", "noisy_circuit", "TrajectoryEstimator"]

_SINGLE_PAULIS = (G.X, G.Y, G.Z)


@dataclass
class DepolarizingNoise:
    """Depolarising noise after every gate, plus optional readout error.

    Parameters
    ----------
    one_qubit:
        Probability that a one-qubit gate is followed by a random Pauli.
    two_qubit:
        Same for two-qubit gates, where a uniformly random non-identity
        two-qubit Pauli is applied.  Set roughly ten times the one-qubit rate
        by default, which is the usual hardware ratio and the reason circuit
        cost is counted in two-qubit gates.
    readout:
        Probability that a measured bit is reported wrong.
    """

    one_qubit: float = 1e-3
    two_qubit: float = 1e-2
    readout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("one_qubit", "two_qubit", "readout"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability, got {value}")

    def scaled(self, factor: float) -> "DepolarizingNoise":
        """The same model with every rate multiplied by ``factor``."""
        return DepolarizingNoise(
            min(1.0, self.one_qubit * factor),
            min(1.0, self.two_qubit * factor),
            self.readout,
        )

    @property
    def is_noiseless(self) -> bool:
        return self.one_qubit == 0.0 and self.two_qubit == 0.0


def noisy_circuit(
    circuit: Circuit,
    model: DepolarizingNoise,
    rng: np.random.Generator,
) -> Circuit:
    """One noise trajectory: the circuit with random Paulis inserted.

    Each gate is followed, with its channel's probability, by a uniformly
    random non-identity Pauli on the qubits it touched.
    """
    out = Circuit(circuit.num_qubits, name=f"{circuit.name}:noisy")
    for instruction in circuit.instructions:
        out.instructions.append(instruction)
        if instruction.gate is None:
            continue
        qubits = instruction.qubits
        if len(qubits) == 1:
            if rng.random() < model.one_qubit:
                out.append(_SINGLE_PAULIS[rng.integers(3)], qubits[0])
        else:
            if rng.random() < model.two_qubit:
                # One of the 15 non-identity two-qubit Paulis, uniformly.
                choice = int(rng.integers(1, 16))
                for offset, qubit in enumerate(qubits[:2]):
                    kind = (choice >> (2 * offset)) & 0b11
                    if kind:
                        out.append(_SINGLE_PAULIS[kind - 1], qubit)
    return out


class TrajectoryEstimator(Estimator):
    """Energy under a Pauli noise model, averaged over trajectories."""

    name = "trajectory"

    def __init__(
        self,
        hamiltonian: PauliOperator,
        model: DepolarizingNoise,
        trajectories: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(hamiltonian)
        self.model = model
        self.trajectories = trajectories
        self.rng = np.random.default_rng(seed)
        self.last_stderr: float = 0.0

    def energy(self, circuit: Circuit) -> float:
        self.evaluations += 1
        if self.model.is_noiseless:
            state = circuit.run().statevector.data
            self.last_stderr = 0.0
            return float(np.real(np.vdot(state, self.hamiltonian.apply(state))))

        samples = np.empty(self.trajectories)
        for index in range(self.trajectories):
            state = noisy_circuit(circuit, self.model, self.rng).run().statevector.data
            samples[index] = float(np.real(np.vdot(state, self.hamiltonian.apply(state))))
        self.last_stderr = float(np.std(samples, ddof=1) / np.sqrt(self.trajectories))
        return float(np.mean(samples))
