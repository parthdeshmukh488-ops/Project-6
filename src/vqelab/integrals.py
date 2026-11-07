"""Molecular integrals, from PySCF into a spin-orbital basis.

Division of labour
------------------
Computing one- and two-electron integrals over Gaussian basis functions is
solved classical quantum chemistry -- a well-optimised, thoroughly tested
piece of numerics that says nothing about quantum computing.  PySCF does that
part.  Everything downstream of this module -- the second-quantised
Hamiltonian, the fermion-to-qubit mapping, the ansatz, the optimiser, the
error mitigation -- is written here from scratch.

What comes out
--------------
PySCF works in *spatial* molecular orbitals; a qubit Hamiltonian needs *spin*
orbitals, since each spin orbital becomes one qubit.  This module does that
expansion, using the interleaved convention

    spin orbital 2p   = spatial orbital p, spin up
    spin orbital 2p+1 = spatial orbital p, spin down

so that the Hartree-Fock reference state is simply the lowest
``num_electrons`` spin orbitals occupied -- which keeps state preparation a
row of ``X`` gates instead of a lookup table.

Integral conventions are the usual trap.  PySCF returns two-electron
integrals in *chemist* notation ``(pq|rs)``, meaning ``p,q`` belong to
electron 1 and ``r,s`` to electron 2.  The Hamiltonian below is written to
match that directly, and the whole construction is checked against PySCF's
own FCI energy, which catches any convention error immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["MolecularData", "load_molecule", "GEOMETRIES"]


#: Equilibrium geometries, in angstroms.
GEOMETRIES: Dict[str, str] = {
    "H2": "H 0 0 0; H 0 0 0.735",
    "LiH": "Li 0 0 0; H 0 0 1.595",
    "BeH2": "Be 0 0 0; H 0 0 1.330; H 0 0 -1.330",
    "H2O": "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
}


@dataclass
class MolecularData:
    """Everything the quantum side needs about a molecule."""

    name: str
    geometry: str
    basis: str
    num_spatial_orbitals: int
    num_electrons: int
    nuclear_repulsion: float
    one_body: np.ndarray          # (nmo, nmo) spatial MO basis
    two_body: np.ndarray          # (nmo,)*4 chemist notation (pq|rs)
    hf_energy: float
    fci_energy: Optional[float] = None
    orbital_energies: Optional[np.ndarray] = None
    frozen_core: int = 0
    active_space: Optional[Tuple[int, int]] = None

    @property
    def num_spin_orbitals(self) -> int:
        return 2 * self.num_spatial_orbitals

    @property
    def num_qubits(self) -> int:
        """One qubit per spin orbital, under Jordan-Wigner."""
        return self.num_spin_orbitals

    def hartree_fock_occupation(self) -> List[int]:
        """Spin orbitals occupied in the Hartree-Fock reference.

        With the interleaved ordering this is just the lowest
        ``num_electrons`` indices.
        """
        return list(range(self.num_electrons))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        extra = "" if self.fci_energy is None else f", FCI {self.fci_energy:.6f}"
        return (
            f"MolecularData({self.name}/{self.basis}, {self.num_qubits} qubits, "
            f"{self.num_electrons} electrons, HF {self.hf_energy:.6f}{extra})"
        )


def load_molecule(
    name: str,
    basis: str = "sto-3g",
    geometry: Optional[str] = None,
    bond_length: Optional[float] = None,
    run_fci: bool = True,
    frozen_core: int = 0,
    active_orbitals: Optional[int] = None,
    charge: int = 0,
    spin: int = 0,
) -> MolecularData:
    """Run Hartree-Fock and return integrals in the molecular-orbital basis.

    Parameters
    ----------
    bond_length:
        For the diatomics, overrides the equilibrium separation -- this is how
        dissociation curves are generated.
    frozen_core:
        Freeze this many lowest spatial orbitals: they stay doubly occupied,
        their energy is folded into the constant, and their effect on the rest
        enters as an extra mean-field potential.  For LiH this removes the
        chemically inert lithium 1s and saves 2 qubits for almost no error.
    active_orbitals:
        Keep only this many spatial orbitals after freezing, discarding the
        highest virtuals.  Reducing the qubit count is the only way heavier
        molecules fit at all.
    """
    from pyscf import ao2mo, fci, gto, scf

    if geometry is None:
        if name not in GEOMETRIES:
            raise ValueError(
                f"unknown molecule {name!r}; pass geometry= explicitly "
                f"or use one of {sorted(GEOMETRIES)}"
            )
        geometry = GEOMETRIES[name]
        if bond_length is not None:
            atoms = [a.strip() for a in geometry.split(";")]
            if len(atoms) != 2:
                raise ValueError(f"bond_length only applies to diatomics, not {name}")
            first = atoms[0].split()[0]
            second = atoms[1].split()[0]
            geometry = f"{first} 0 0 0; {second} 0 0 {bond_length}"

    mol = gto.M(atom=geometry, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol).run()
    if not mf.converged:                      # pragma: no cover - rare
        raise RuntimeError(f"Hartree-Fock did not converge for {name}")

    coefficients = mf.mo_coeff
    nmo_total = coefficients.shape[1]

    fci_energy = None
    if run_fci:
        fci_energy = float(fci.FCI(mf).kernel()[0])

    # --- active space -------------------------------------------------
    stop = nmo_total if active_orbitals is None else frozen_core + active_orbitals
    if stop > nmo_total:
        raise ValueError(
            f"requested {active_orbitals} active orbitals after freezing "
            f"{frozen_core}, but the basis has only {nmo_total}"
        )
    active = slice(frozen_core, stop)

    hcore = coefficients.T @ mf.get_hcore() @ coefficients
    eri_full = ao2mo.restore(1, ao2mo.kernel(mol, coefficients), nmo_total)

    constant = float(mol.energy_nuc())
    if frozen_core:
        # Frozen orbitals contribute a fixed energy, plus a mean-field
        # (Coulomb minus exchange) potential felt by the active electrons.
        core = slice(0, frozen_core)
        for i in range(frozen_core):
            constant += 2.0 * hcore[i, i]
            for j in range(frozen_core):
                constant += 2.0 * eri_full[i, i, j, j] - eri_full[i, j, j, i]
        effective = hcore.copy()
        for p in range(nmo_total):
            for q in range(nmo_total):
                for i in range(frozen_core):
                    effective[p, q] += (
                        2.0 * eri_full[p, q, i, i] - eri_full[p, i, i, q]
                    )
        hcore = effective
        del core

    one_body = np.ascontiguousarray(hcore[active, active])
    two_body = np.ascontiguousarray(eri_full[active, active, active, active])
    num_electrons = mol.nelectron - 2 * frozen_core

    return MolecularData(
        name=name,
        geometry=geometry,
        basis=basis,
        num_spatial_orbitals=one_body.shape[0],
        num_electrons=num_electrons,
        nuclear_repulsion=constant,
        one_body=one_body,
        two_body=two_body,
        hf_energy=float(mf.e_tot),
        fci_energy=fci_energy,
        orbital_energies=np.asarray(mf.mo_energy),
        frozen_core=frozen_core,
        active_space=(num_electrons, one_body.shape[0]),
    )
