# vqelab — VQE for molecules, on my own simulator

Variational quantum eigensolver built from first principles and run on
[`../simulator`](../simulator), the state-vector engine from the first project
in this repository.

PySCF supplies the molecular integrals — computing Gaussian integrals is
solved classical chemistry and says nothing about quantum computing.
**Everything downstream of them is written here**: the second-quantised
Hamiltonian, the Jordan–Wigner mapping, the ansatz circuits, measurement
grouping, the optimiser and its gradients.

**Status: core complete.** Dissociation curves, noise and error mitigation
are in progress.

## Validated against classical chemistry

| check | result |
|---|---|
| Hartree–Fock energy from the qubit Hamiltonian (H₂) | matches PySCF to **2 × 10⁻¹⁶ Ha** |
| Ground state of the qubit Hamiltonian (H₂) | matches PySCF FCI to **4 × 10⁻¹⁶ Ha** |
| LiH, 12 qubits, 631 Pauli terms | matches FCI to **1.5 × 10⁻¹⁴ Ha** |
| VQE with UCCSD (H₂) | **7 × 10⁻¹⁵ Ha** error, 100% of correlation energy |
| Frozen-core approximation (LiH, BeH₂, H₂O) | 0.8–3.4 × 10⁻⁴ Ha, inside chemical accuracy |

That FCI agreement is the load-bearing test: it exercises the integral
handling, the spin-orbital expansion, the operator-ordering conventions and
the fermion-to-qubit mapping simultaneously, against a completely independent
implementation of the same physics.

## The pieces

| module | what it does |
|---|---|
| `pauli.py` | Pauli operators in symplectic form — two integers per string, so multiplication is XOR plus a parity count |
| `integrals.py` | PySCF → spin orbitals, with frozen-core and active-space reduction |
| `hamiltonian.py` | Jordan–Wigner: `Z` strings that reproduce fermionic anticommutation on commuting qubits |
| `ansatz.py` | Pauli-exponential circuits, hardware-efficient layers, UCCSD |
| `estimator.py` | exact and shot-based energies, with qubit-wise commuting measurement grouping |
| `vqe.py` | the optimisation loop, parameter-shift and finite-difference gradients |
| `exact.py` | matrix-free Lanczos reference energies |

## Things worth pointing at

**Jordan–Wigner is tested, not assumed.** Electrons anticommute; qubits do
not. The `Z` strings exist to recover the minus signs, and the test suite
checks `{aᵢ, aⱼ†} = δᵢⱼ` and `{aᵢ, aⱼ} = 0` for every pair, as matrices.

**The variational principle is a test.** No trial state may ever report an
energy below the true ground state. If one does, something is broken — it
cannot be a lucky result. Checked over random parameters for both ansätze.

**Brillouin's theorem falls out.** Single excitations alone cannot improve on
Hartree–Fock, because HF orbitals are chosen precisely so their gradient
vanishes. The singles-only ansatz reproduces that.

**Measurement grouping.** Qubit-wise commuting Pauli strings share a
measurement basis: 14 → 5 settings for H₂, 275 → 75 for LiH, 550 → 178 for
H₂O. Roughly a 3× cut in the dominant practical cost.

**The parameter-shift rule is exact**, not finite differences with a good step
size — the energy's dependence on a single Pauli rotation angle is a pure
sinusoid. Verified against central differences to 4 × 10⁻¹¹.

## Running it

```bash
pip install -e ../simulator
pip install -e ".[dev,plot]"
pytest tests/ -q
```

```python
from vqelab import load_molecule, fermionic_hamiltonian
from vqelab.ansatz import UCCSDAnsatz
from vqelab.vqe import VQE

mol = load_molecule("H2")
H = fermionic_hamiltonian(mol)
result = VQE(H, UCCSDAnsatz(mol)).run(
    reference_energy=mol.fci_energy, hartree_fock_energy=mol.hf_energy
)
print(result.summary())
```

## Correctness

138 tests, 96% coverage.

## Still to come

- Dissociation curves for H₂ and LiH, where Hartree–Fock breaks down and
  correlation matters most
- Noise models and **zero-noise extrapolation**
- Readout-error mitigation
- Barren-plateau measurements for the hardware-efficient ansatz

## License

MIT
