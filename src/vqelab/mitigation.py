"""Error mitigation: removing the bias noise puts on an expectation value.

Error *correction* fixes errors as they happen, needs many physical qubits
per logical one, and is years away at useful scale (see ``../qec``).  Error
*mitigation* does something much cheaper and much weaker: it leaves the
errors in place and removes their effect from the final number, statistically,
by running deliberately altered circuits.  It buys no coherence, only
accuracy on an average -- which is exactly what VQE needs.

Zero-noise extrapolation
------------------------
The one idea: you cannot make the noise smaller, but you can make it *bigger*
on purpose, measure how the energy moves, and extrapolate back to zero.

Noise is amplified here by **unitary folding** -- replacing the circuit ``C``
with ``C (C^dag C)^n``.  That is the identity mathematically, so the ideal
answer is unchanged, while the gate count, and therefore the accumulated
error, grows by a factor of ``2n + 1``.  It needs no access to the hardware's
error rates, which is why real experiments use it.

Then fit ``E(lambda)`` and evaluate at ``lambda = 0``.  The fit's form is an
assumption, and a wrong one shows up as a bad extrapolation, so linear,
polynomial (Richardson) and exponential models are all available and the
residual is reported.

Readout mitigation
------------------
Measurement error is special: it acts only at the very end and is
characterised completely by preparing each basis state and seeing what comes
back.  That gives a calibration matrix, and inverting it undoes the damage.
The inverse can produce slightly negative "probabilities", so the result is
projected back onto the simplex rather than used raw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qvolve import Circuit

__all__ = [
    "fold_global",
    "ZNEResult",
    "zero_noise_extrapolation",
    "ReadoutMitigator",
]


def fold_global(circuit: Circuit, scale: int) -> Circuit:
    """Return ``C (C^dag C)^n`` with ``scale = 2n + 1``.

    Mathematically the identity, so the noiseless expectation value is
    untouched; physically it multiplies the gate count, and therefore the
    accumulated error, by ``scale``.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"global folding needs an odd scale >= 1, got {scale}")
    folded = Circuit(circuit.num_qubits, name=f"{circuit.name}:fold{scale}")
    folded.compose(circuit)
    inverse = circuit.inverse()
    for _ in range((scale - 1) // 2):
        folded.compose(inverse)
        folded.compose(circuit)
    return folded


@dataclass
class ZNEResult:
    """Outcome of an extrapolation."""

    mitigated: float
    scales: List[float]
    energies: List[float]
    stderrs: List[float] = field(default_factory=list)
    model: str = "linear"
    coefficients: Optional[np.ndarray] = None
    residual: float = 0.0

    @property
    def unmitigated(self) -> float:
        """The energy at the smallest noise scale actually measured."""
        return self.energies[int(np.argmin(self.scales))]

    def improvement(self, exact: float) -> Dict[str, float]:
        """How much of the noise-induced error the extrapolation removed."""
        before = abs(self.unmitigated - exact)
        after = abs(self.mitigated - exact)
        return {
            "error_before": before,
            "error_after": after,
            "factor": before / after if after > 0 else float("inf"),
            "removed_fraction": 1.0 - after / before if before > 0 else 0.0,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ZNEResult({self.model}, unmitigated={self.unmitigated:.6f}, "
            f"mitigated={self.mitigated:.6f})"
        )


def zero_noise_extrapolation(
    measure: Callable[[float], float],
    scales: Sequence[float] = (1.0, 3.0, 5.0),
    model: str = "linear",
    stderrs: Optional[Sequence[float]] = None,
) -> ZNEResult:
    """Measure the energy at several noise scales and extrapolate to zero.

    Parameters
    ----------
    measure:
        Called with a noise scale, returns the energy at that scale.
    model:
        ``"linear"``, ``"richardson"`` (a polynomial through every point) or
        ``"exponential"`` (``a + b exp(-c lambda)``, which respects the fact
        that an infinitely noisy circuit gives a finite, bounded answer rather
        than diverging).
    """
    scales = [float(s) for s in scales]
    energies = [float(measure(s)) for s in scales]
    x = np.asarray(scales)
    y = np.asarray(energies)

    if model == "linear":
        coefficients = np.polyfit(x, y, 1)
        mitigated = float(np.polyval(coefficients, 0.0))
        residual = float(np.sqrt(np.mean((np.polyval(coefficients, x) - y) ** 2)))
    elif model == "richardson":
        # A polynomial of degree n-1 through all n points, evaluated at zero.
        degree = len(x) - 1
        coefficients = np.polyfit(x, y, degree)
        mitigated = float(np.polyval(coefficients, 0.0))
        residual = 0.0                      # interpolation: exact by construction
    elif model == "exponential":
        import warnings

        from scipy.optimize import OptimizeWarning, curve_fit

        def form(lam, a, b, c):
            return a + b * np.exp(-c * lam)

        guess = [y[-1], y[0] - y[-1], 0.5]
        try:
            # Three points and three parameters means the covariance is
            # undetermined; SciPy warns about that, and it is expected rather
            # than a problem -- the fit itself is exact interpolation.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                coefficients, _ = curve_fit(form, x, y, p0=guess, maxfev=20_000)
            mitigated = float(form(0.0, *coefficients))
            residual = float(np.sqrt(np.mean((form(x, *coefficients) - y) ** 2)))
        except Exception:                   # pragma: no cover - fit failure
            coefficients = np.polyfit(x, y, 1)
            mitigated = float(np.polyval(coefficients, 0.0))
            residual = float("nan")
            model = "linear (exponential fit failed)"
    else:
        raise ValueError(f"unknown extrapolation model {model!r}")

    return ZNEResult(
        mitigated=mitigated,
        scales=scales,
        energies=energies,
        stderrs=list(stderrs or []),
        model=model,
        coefficients=np.asarray(coefficients),
        residual=residual,
    )


class ReadoutMitigator:
    """Undo measurement error by inverting a calibration matrix.

    Calibration: prepare each of the ``2^n`` basis states, measure, and record
    where the counts actually landed.  That matrix ``A`` maps true
    distributions to observed ones, so ``A^-1`` undoes it.

    This is exact in principle and exponential in practice -- ``2^n``
    calibration circuits -- so it is used here on small registers to
    demonstrate the effect, with tensored or matrix-free variants being what
    real experiments use.
    """

    def __init__(self, num_qubits: int, readout_error: float, seed: Optional[int] = None):
        if num_qubits > 12:
            raise ValueError(
                f"full calibration needs 2^{num_qubits} circuits; use a "
                "tensored approximation above 12 qubits"
            )
        self.num_qubits = num_qubits
        self.readout_error = readout_error
        self.rng = np.random.default_rng(seed)
        self.matrix = self._calibrate()
        self.inverse = np.linalg.pinv(self.matrix)

    def _calibrate(self) -> np.ndarray:
        """Exact calibration matrix for independent per-qubit bit flips."""
        p = self.readout_error
        single = np.array([[1 - p, p], [p, 1 - p]])
        matrix = np.array([[1.0]])
        for _ in range(self.num_qubits):
            matrix = np.kron(matrix, single)
        return matrix

    def apply_noise(self, probabilities: np.ndarray) -> np.ndarray:
        """What a noisy readout would report for a true distribution."""
        return self.matrix @ np.asarray(probabilities, dtype=float)

    def mitigate(self, observed: np.ndarray) -> np.ndarray:
        """Invert the calibration, then project back onto valid probabilities.

        Inversion amplifies statistical noise and can push entries slightly
        negative, which is not a distribution.  Clipping and renormalising is
        the standard, honest repair.
        """
        corrected = self.inverse @ np.asarray(observed, dtype=float)
        corrected = np.clip(corrected, 0.0, None)
        total = corrected.sum()
        return corrected / total if total > 0 else corrected
