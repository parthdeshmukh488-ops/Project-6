"""The variational quantum eigensolver.

The whole algorithm rests on one line of linear algebra -- the variational
principle:

    <psi|H|psi> >= E_0   for every normalised |psi>

So *any* trial state gives an upper bound on the ground-state energy, and
minimising over a parameterised family can only improve it.  There is no way
to accidentally report an energy below the truth; if you do, something is
broken.  (The tests check exactly that.)

The quantum computer's only job is preparing ``|psi(theta)>`` and measuring
``<H>``.  A classical optimiser drives ``theta``.  That split is what makes
VQE plausible on near-term hardware: the circuit stays shallow, and no
long-lived coherence is needed.

What makes it hard
------------------
Chemical accuracy -- 1.6 millihartree, the threshold where predicted reaction
rates become useful -- is about 0.1% of a molecule's total energy, and
Hartree-Fock already gets ~99%.  VQE is chasing the last fraction of a
percent, which means the measurement precision, the ansatz expressiveness and
the optimiser all have to be good simultaneously.

Gradients
---------
Where every parameter drives a single Pauli rotation, the **parameter-shift
rule** gives an exact derivative from two more energy evaluations:

    dE/dtheta = [E(theta + pi/2) - E(theta - pi/2)] / 2

This is not finite differences with a lucky step size -- it is exact, because
the energy's dependence on a single Pauli rotation angle is a pure sinusoid.
For UCCSD a single parameter drives several rotations with different
coefficients, so the simple rule does not apply and central differences are
used instead; the ansatz declares which it supports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from qvolve import Circuit

from .ansatz import Ansatz, HardwareEfficientAnsatz, UCCSDAnsatz
from .estimator import Estimator, StatevectorEstimator
from .hamiltonian import number_operator, spin_z_operator
from .pauli import PauliOperator

__all__ = ["VQE", "VQEResult", "CHEMICAL_ACCURACY"]

#: Chemical accuracy in hartree -- the error below which predicted reaction
#: rates are reliable at room temperature.  The target every VQE paper quotes.
CHEMICAL_ACCURACY = 1.6e-3


@dataclass
class VQEResult:
    """Outcome of one optimisation."""

    energy: float
    parameters: np.ndarray
    iterations: int
    evaluations: int
    seconds: float
    converged: bool
    message: str = ""
    history: List[float] = field(default_factory=list)
    reference_energy: Optional[float] = None
    hartree_fock_energy: Optional[float] = None
    particle_number: Optional[float] = None
    spin_z: Optional[float] = None

    @property
    def error(self) -> Optional[float]:
        if self.reference_energy is None:
            return None
        return self.energy - self.reference_energy

    @property
    def chemically_accurate(self) -> Optional[bool]:
        if self.error is None:
            return None
        return abs(self.error) < CHEMICAL_ACCURACY

    @property
    def correlation_recovered(self) -> Optional[float]:
        """Fraction of the correlation energy the ansatz captured.

        Hartree-Fock already gets ~99% of the total energy; the interesting
        quantity is how much of what it *misses* was recovered.
        """
        if self.reference_energy is None or self.hartree_fock_energy is None:
            return None
        total = self.hartree_fock_energy - self.reference_energy
        if abs(total) < 1e-12:
            return 1.0
        return (self.hartree_fock_energy - self.energy) / total

    def summary(self) -> str:
        lines = [f"energy        {self.energy:.10f} Ha"]
        if self.reference_energy is not None:
            lines.append(f"reference     {self.reference_energy:.10f} Ha")
            lines.append(
                f"error         {self.error:+.3e} Ha"
                f"  ({'within' if self.chemically_accurate else 'OUTSIDE'} "
                "chemical accuracy)"
            )
        if self.correlation_recovered is not None:
            lines.append(f"correlation   {self.correlation_recovered * 100:.2f}% recovered")
        lines.append(
            f"optimiser     {self.iterations} iterations, "
            f"{self.evaluations} energy evaluations, {self.seconds:.2f}s"
        )
        if self.particle_number is not None:
            lines.append(
                f"symmetry      <N> = {self.particle_number:.6f}, "
                f"<Sz> = {self.spin_z:+.6f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"VQEResult(energy={self.energy:.8f}, error={self.error})"


class VQE:
    """Drive a classical optimiser over an ansatz's parameters."""

    def __init__(
        self,
        hamiltonian: PauliOperator,
        ansatz: Ansatz,
        estimator: Optional[Estimator] = None,
        optimizer: str = "L-BFGS-B",
        gradient: str = "auto",
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.estimator = estimator or StatevectorEstimator(hamiltonian)
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.callback = callback
        self.history: List[float] = []

        if gradient == "auto":
            # The shift rule needs each parameter to drive exactly one Pauli
            # rotation, which holds for hardware-efficient layers but not for
            # UCCSD, where one amplitude drives a whole cluster of them.
            gradient = (
                "parameter-shift"
                if isinstance(ansatz, HardwareEfficientAnsatz)
                else "finite-difference"
            )
        if gradient not in ("parameter-shift", "finite-difference", "none"):
            raise ValueError(f"unknown gradient method {gradient!r}")
        self.gradient_method = gradient

    # ------------------------------------------------------------------
    def energy(self, parameters: Sequence[float]) -> float:
        return self.estimator.energy(self.ansatz.circuit(parameters))

    def gradient(self, parameters: Sequence[float]) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=float)
        grad = np.zeros_like(parameters)
        if self.gradient_method == "parameter-shift":
            shift = np.pi / 2
            for index in range(parameters.size):
                plus = parameters.copy()
                minus = parameters.copy()
                plus[index] += shift
                minus[index] -= shift
                grad[index] = 0.5 * (self.energy(plus) - self.energy(minus))
        else:
            step = 1e-5
            for index in range(parameters.size):
                plus = parameters.copy()
                minus = parameters.copy()
                plus[index] += step
                minus[index] -= step
                grad[index] = (self.energy(plus) - self.energy(minus)) / (2 * step)
        return grad

    # ------------------------------------------------------------------
    def run(
        self,
        initial_parameters: Optional[Sequence[float]] = None,
        reference_energy: Optional[float] = None,
        hartree_fock_energy: Optional[float] = None,
    ) -> VQEResult:
        from scipy.optimize import minimize

        if initial_parameters is None:
            initial_parameters = self.ansatz.initial_parameters()
        x0 = np.asarray(initial_parameters, dtype=float)

        self.history = []
        self.estimator.reset_counter()

        def objective(parameters):
            value = self.energy(parameters)
            self.history.append(value)
            if self.callback is not None:
                self.callback(len(self.history), value)
            return value

        gradient_free = self.optimizer in ("COBYLA", "Nelder-Mead", "Powell")
        jac = None if (gradient_free or self.gradient_method == "none") else self.gradient

        options: Dict[str, object] = {"maxiter": self.max_iterations}
        if self.optimizer == "COBYLA":
            options = {"maxiter": self.max_iterations, "tol": self.tolerance}

        start = time.perf_counter()
        result = minimize(
            objective, x0, method=self.optimizer, jac=jac,
            tol=self.tolerance, options=options,
        )
        seconds = time.perf_counter() - start

        best = float(min(self.history)) if self.history else float(result.fun)
        parameters = np.asarray(result.x, dtype=float)

        particle_number = spin = None
        if isinstance(self.estimator, StatevectorEstimator):
            state = self.estimator.state(self.ansatz.circuit(parameters))
            n_op = number_operator(self.hamiltonian.num_qubits)
            particle_number = float(np.real(np.vdot(state, n_op.apply(state))))
            if self.hamiltonian.num_qubits % 2 == 0:
                sz_op = spin_z_operator(self.hamiltonian.num_qubits)
                spin = float(np.real(np.vdot(state, sz_op.apply(state))))

        return VQEResult(
            energy=best,
            parameters=parameters,
            iterations=int(getattr(result, "nit", len(self.history))),
            evaluations=self.estimator.evaluations,
            seconds=seconds,
            converged=bool(result.success),
            message=str(result.message),
            history=list(self.history),
            reference_energy=reference_energy,
            hartree_fock_energy=hartree_fock_energy,
            particle_number=particle_number,
            spin_z=spin,
        )
