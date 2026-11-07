"""Pauli operators in the symplectic representation.

A molecular Hamiltonian, once mapped to qubits, is a weighted sum of Pauli
strings -- often hundreds or thousands of them.  Everything downstream
(measuring an energy, exponentiating a term, diagonalising exactly) is built
on multiplying and adding those strings, so the representation matters.

Storing them as text like ``"XIZY"`` makes multiplication a character-by-
character special case.  Instead each string is two integers:

    P = X^x Z^z     with x, z bit masks over the qubits

Qubit ``q`` owns bit ``n - 1 - q`` of each mask -- qubit 0 is the *most*
significant bit, matching the big-endian convention of the ``qvolve``
simulator these operators are measured on, and matching the order in which
``numpy.kron`` builds a tensor product.  Both bits set on a qubit means
``XZ``, which is ``-iY``.  Then
multiplication is three machine instructions,

    (X^x1 Z^z1)(X^x2 Z^z2) = (-1)^{z1 . x2} X^{x1 xor x2} Z^{z1 xor z2}

because ``ZX = -XZ`` on each qubit where both appear, and the whole
bookkeeping reduces to XOR and a parity count.  This is the same symplectic
encoding used for stabiliser codes in the error-correction project next door,
for the same reason.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

__all__ = ["PauliOperator", "pauli_matrix", "qubit_bit"]

_SINGLE = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(label: str) -> np.ndarray:
    """Dense matrix of a Pauli string, qubit 0 leftmost."""
    out = np.array([[1.0 + 0j]])
    for char in label:
        out = np.kron(out, _SINGLE[char])
    return out


def _parity(value: int) -> int:
    return bin(value).count("1") & 1


def qubit_bit(num_qubits: int, qubit: int) -> int:
    """Mask with only ``qubit``'s bit set, big-endian (qubit 0 is the MSB)."""
    if not 0 <= qubit < num_qubits:
        raise ValueError(f"qubit {qubit} out of range for {num_qubits} qubits")
    return 1 << (num_qubits - 1 - qubit)


class PauliOperator:
    """A complex linear combination of Pauli strings on ``num_qubits`` qubits.

    Terms are keyed by ``(x, z)`` bit masks, so identical strings coalesce
    automatically on construction and addition.
    """

    __slots__ = ("num_qubits", "terms")

    def __init__(
        self,
        num_qubits: int,
        terms: Optional[Dict[Tuple[int, int], complex]] = None,
    ) -> None:
        self.num_qubits = int(num_qubits)
        self.terms: Dict[Tuple[int, int], complex] = dict(terms or {})

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def identity(cls, num_qubits: int, coefficient: complex = 1.0) -> "PauliOperator":
        return cls(num_qubits, {(0, 0): complex(coefficient)})

    @classmethod
    def zero(cls, num_qubits: int) -> "PauliOperator":
        return cls(num_qubits, {})

    @classmethod
    def from_label(cls, label: str, coefficient: complex = 1.0) -> "PauliOperator":
        """Build from a string like ``"XIZY"``, qubit 0 leftmost.

        ``Y`` carries a factor of ``i`` because ``Y = i X Z``, and that phase
        is folded into the coefficient so the stored form stays a pure
        ``X^x Z^z``.
        """
        n = len(label)
        x = z = 0
        phase = 1.0 + 0j
        for qubit, char in enumerate(label.upper()):
            bit = qubit_bit(n, qubit)
            if char == "X":
                x |= bit
            elif char == "Z":
                z |= bit
            elif char == "Y":
                x |= bit
                z |= bit
                phase *= 1j
            elif char != "I":
                raise ValueError(f"invalid Pauli character {char!r}")
        return cls(n, {(x, z): complex(coefficient) * phase})

    # ------------------------------------------------------------------
    # algebra
    # ------------------------------------------------------------------
    def _check(self, other: "PauliOperator") -> None:
        if self.num_qubits != other.num_qubits:
            raise ValueError(
                f"qubit counts differ: {self.num_qubits} vs {other.num_qubits}"
            )

    def __add__(self, other: "PauliOperator") -> "PauliOperator":
        self._check(other)
        terms = dict(self.terms)
        for key, value in other.terms.items():
            total = terms.get(key, 0j) + value
            if total == 0:
                terms.pop(key, None)
            else:
                terms[key] = total
        return PauliOperator(self.num_qubits, terms)

    def __sub__(self, other: "PauliOperator") -> "PauliOperator":
        return self + (other * -1.0)

    def __neg__(self) -> "PauliOperator":
        return self * -1.0

    def __mul__(self, other) -> "PauliOperator":
        if isinstance(other, PauliOperator):
            return self._compose(other)
        return PauliOperator(
            self.num_qubits, {k: v * other for k, v in self.terms.items()}
        )

    __rmul__ = __mul__

    def _compose(self, other: "PauliOperator") -> "PauliOperator":
        """Operator product, term by term.

        The only subtlety is the sign: commuting ``Z^z1`` past ``X^x2`` picks
        up ``(-1)`` for every qubit where both act, which is the parity of
        ``z1 & x2``.
        """
        self._check(other)
        terms: Dict[Tuple[int, int], complex] = {}
        for (x1, z1), c1 in self.terms.items():
            for (x2, z2), c2 in other.terms.items():
                sign = -1.0 if _parity(z1 & x2) else 1.0
                key = (x1 ^ x2, z1 ^ z2)
                total = terms.get(key, 0j) + sign * c1 * c2
                if total == 0:
                    terms.pop(key, None)
                else:
                    terms[key] = total
        return PauliOperator(self.num_qubits, terms)

    def dagger(self) -> "PauliOperator":
        """Hermitian conjugate.

        ``X^x Z^z`` is Hermitian up to the sign from reversing the order of
        the ``X`` and ``Z`` factors on qubits carrying both, so conjugating is
        a coefficient conjugation plus that parity sign.
        """
        terms = {}
        for (x, z), c in self.terms.items():
            sign = -1.0 if _parity(x & z) else 1.0
            terms[(x, z)] = np.conj(c) * sign
        return PauliOperator(self.num_qubits, terms)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def simplify(self, tolerance: float = 1e-12) -> "PauliOperator":
        """Drop terms whose coefficient is numerically zero."""
        return PauliOperator(
            self.num_qubits,
            {k: v for k, v in self.terms.items() if abs(v) > tolerance},
        )

    def is_hermitian(self, tolerance: float = 1e-10) -> bool:
        difference = (self - self.dagger()).simplify(tolerance)
        return len(difference.terms) == 0

    @property
    def num_terms(self) -> int:
        return len(self.terms)

    def constant(self) -> complex:
        """The coefficient of the identity string."""
        return self.terms.get((0, 0), 0j)

    def label_terms(self) -> Iterator[Tuple[str, complex]]:
        """Yield ``(label, coefficient)`` with ``Y``s made explicit.

        Each qubit carrying both an ``X`` and a ``Z`` is rewritten as ``Y``,
        which costs a factor of ``-i`` since ``XZ = -iY``.
        """
        for (x, z), coefficient in self.terms.items():
            chars: List[str] = []
            phase = 1.0 + 0j
            for qubit in range(self.num_qubits):
                shift = self.num_qubits - 1 - qubit
                bx = (x >> shift) & 1
                bz = (z >> shift) & 1
                if bx and bz:
                    chars.append("Y")
                    phase *= -1j
                elif bx:
                    chars.append("X")
                elif bz:
                    chars.append("Z")
                else:
                    chars.append("I")
            yield "".join(chars), coefficient * phase

    def weight(self, key: Tuple[int, int]) -> int:
        """Number of qubits a term acts on non-trivially."""
        x, z = key
        return bin(x | z).count("1")

    def max_weight(self) -> int:
        return max((self.weight(k) for k in self.terms), default=0)

    # ------------------------------------------------------------------
    # linear algebra
    # ------------------------------------------------------------------
    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply the operator to a state vector, without forming a matrix.

        Each term acts as a permutation and a sign pattern::

            (X^x Z^z) |b> = (-1)^{popcount(z & b)} |b xor x>

        so a whole term costs one gather and one scaled add -- ``O(2^n)``
        rather than the ``O(4^n)`` a dense matrix-vector product would need.
        This is what makes exact diagonalisation of a 14-qubit Hamiltonian
        with a thousand terms practical.

        Index convention matches the masks: qubit 0 is the most significant
        bit of the basis index, as in ``numpy.kron`` and in ``qvolve``.
        """
        dimension = 1 << self.num_qubits
        vector = np.asarray(vector, dtype=complex).ravel()
        if vector.size != dimension:
            raise ValueError(f"expected {dimension} amplitudes, got {vector.size}")
        index = np.arange(dimension, dtype=np.int64)
        out = np.zeros(dimension, dtype=complex)
        for (x, z), coefficient in self.terms.items():
            source = index ^ x
            parity = _popcount(source & z) & 1
            signs = np.where(parity, -1.0, 1.0)
            out += coefficient * signs * vector[source]
        return out

    def to_matrix(self) -> np.ndarray:
        """Dense matrix.  Only sensible for a handful of qubits."""
        if self.num_qubits > 14:
            raise ValueError(
                f"{self.num_qubits} qubits would need a "
                f"{1 << self.num_qubits} x {1 << self.num_qubits} matrix"
            )
        dimension = 1 << self.num_qubits
        out = np.zeros((dimension, dimension), dtype=complex)
        basis = np.eye(dimension, dtype=complex)
        for column in range(dimension):
            out[:, column] = self.apply(basis[:, column])
        return out

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PauliOperator({self.num_qubits} qubits, {self.num_terms} terms, "
            f"max weight {self.max_weight()})"
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        rows = sorted(self.label_terms(), key=lambda kv: -abs(kv[1]))
        lines = [f"{c.real:+.8f} {'' if abs(c.imag) < 1e-12 else f'{c.imag:+.8f}j '}{l}"
                 for l, c in rows[:20]]
        if len(rows) > 20:
            lines.append(f"... and {len(rows) - 20} more terms")
        return "\n".join(lines)


def _popcount(values: np.ndarray) -> np.ndarray:
    """Vectorised population count for int64 arrays."""
    v = values.astype(np.int64)
    count = np.zeros_like(v)
    while np.any(v):
        count += v & 1
        v >>= 1
    return count
