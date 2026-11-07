"""Pauli algebra tests, against densely built matrices."""

import itertools

import numpy as np
import pytest

from vqelab.pauli import PauliOperator, pauli_matrix, qubit_bit

LABELS_1 = ["I", "X", "Y", "Z"]


@pytest.mark.parametrize("a", LABELS_1)
@pytest.mark.parametrize("b", LABELS_1)
def test_single_qubit_products(a, b):
    product = PauliOperator.from_label(a) * PauliOperator.from_label(b)
    assert np.allclose(product.to_matrix(), pauli_matrix(a) @ pauli_matrix(b))


@pytest.mark.parametrize(
    "a,b",
    [("XY", "ZI"), ("XZ", "YY"), ("YXZ", "ZYX"), ("IXY", "YXI"),
     ("XYZI", "ZIYX"), ("YYYY", "XZXZ"), ("IIII", "XYZI")],
)
def test_multi_qubit_products_match_dense(a, b):
    product = PauliOperator.from_label(a) * PauliOperator.from_label(b)
    assert np.allclose(product.to_matrix(), pauli_matrix(a) @ pauli_matrix(b))


def test_pauli_commutation_relations():
    X = PauliOperator.from_label("X")
    Y = PauliOperator.from_label("Y")
    Z = PauliOperator.from_label("Z")
    assert np.allclose((X * Y).to_matrix(), 1j * Z.to_matrix())
    assert np.allclose((Y * Z).to_matrix(), 1j * X.to_matrix())
    assert np.allclose((Z * X).to_matrix(), 1j * Y.to_matrix())
    assert np.allclose((X * Y + Y * X).to_matrix(), 0)       # anticommute
    for P in (X, Y, Z):
        assert np.allclose((P * P).to_matrix(), np.eye(2))


def test_endianness_matches_kron_and_qvolve():
    """Qubit 0 must be the most significant bit, as numpy.kron builds it."""
    assert qubit_bit(4, 0) == 0b1000
    assert qubit_bit(4, 3) == 0b0001
    for label in ("XZ", "YXZ", "XYZI"):
        assert np.allclose(
            PauliOperator.from_label(label).to_matrix(), pauli_matrix(label)
        )


def test_expectation_agrees_with_the_simulator():
    """Cross-check against qvolve, which is an independent implementation."""
    from qvolve import StateVector

    for label in ("XY", "ZZI", "XYZ", "IYXZ", "ZZZZ"):
        state = StateVector.random(len(label), rng=np.random.default_rng(1))
        mine = float(
            np.real(state.data.conj() @ PauliOperator.from_label(label).apply(state.data))
        )
        assert mine == pytest.approx(state.expectation_pauli(label), abs=1e-12)


def test_addition_and_scaling():
    op = 2.0 * PauliOperator.from_label("XY") + PauliOperator.from_label("XY")
    assert op.num_terms == 1
    assert np.allclose(op.to_matrix(), 3 * pauli_matrix("XY"))
    assert (op - op).simplify().num_terms == 0
    assert np.allclose((-op).to_matrix(), -3 * pauli_matrix("XY"))


def test_identity_and_zero():
    identity = PauliOperator.identity(3)
    assert np.allclose(identity.to_matrix(), np.eye(8))
    assert PauliOperator.zero(3).num_terms == 0
    assert identity.constant() == 1.0


def test_hermiticity():
    hermitian = (
        0.5 * PauliOperator.from_label("XY")
        + 1.5 * PauliOperator.from_label("ZZ")
        - 0.25 * PauliOperator.from_label("IY")
    )
    assert hermitian.is_hermitian()
    assert np.allclose(hermitian.to_matrix(), hermitian.to_matrix().conj().T)
    assert not (1j * PauliOperator.from_label("XY")).is_hermitian()


def test_dagger_matches_conjugate_transpose():
    rng = np.random.default_rng(0)
    op = PauliOperator(3)
    for label in ("XYZ", "ZZI", "IYX", "XXX"):
        op = op + (complex(*rng.normal(size=2))) * PauliOperator.from_label(label)
    assert np.allclose(op.dagger().to_matrix(), op.to_matrix().conj().T)


def test_apply_matches_matrix_multiplication():
    rng = np.random.default_rng(2)
    op = (
        0.3 * PauliOperator.from_label("XYZ")
        - 1.1 * PauliOperator.from_label("ZZI")
        + 0.7 * PauliOperator.from_label("IXX")
    )
    vector = rng.normal(size=8) + 1j * rng.normal(size=8)
    assert np.allclose(op.apply(vector), op.to_matrix() @ vector)


def test_label_round_trip():
    for label in ("XYZI", "IIII", "ZZZZ", "YYXX"):
        op = PauliOperator.from_label(label, 2.5)
        (recovered, coefficient), = op.label_terms()
        assert recovered == label
        assert coefficient == pytest.approx(2.5)


def test_weight_accounting():
    op = PauliOperator.from_label("XIZI") + PauliOperator.from_label("XYZI")
    assert op.max_weight() == 3
    assert sorted(op.weight(k) for k in op.terms) == [2, 3]


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        PauliOperator.from_label("XQ")
    with pytest.raises(ValueError):
        PauliOperator.from_label("XY") + PauliOperator.from_label("XYZ")
    with pytest.raises(ValueError):
        PauliOperator.from_label("XY").apply(np.zeros(8))
    with pytest.raises(ValueError):
        qubit_bit(3, 5)
