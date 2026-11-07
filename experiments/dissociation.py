"""Dissociation curves: where Hartree-Fock fails and correlation matters.

Pulling a molecule apart is the classic stress test for electronic structure.
Near equilibrium Hartree-Fock is already good, and VQE has little to prove.
Stretch the bond and HF collapses: a restricted single determinant forces the
two electrons to stay paired in the same spatial orbital, so at large
separation it describes a superposition of H-H and H+/H- instead of two
neutral atoms, and the energy comes out far too high.

Exactly there, static correlation dominates, and a method that captures it
tracks the true curve all the way out.  Reproducing that failure *and* its
repair is the real test of a chemistry stack.

    python experiments/dissociation.py [--molecule H2|LiH] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqelab import fermionic_hamiltonian, load_molecule
from vqelab.ansatz import UCCSDAnsatz
from vqelab.vqe import CHEMICAL_ACCURACY, VQE

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

GRIDS = {
    "H2": np.round(np.arange(0.30, 2.81, 0.125), 4),
    "LiH": np.round(np.arange(1.0, 3.21, 0.2), 4),
}
FROZEN = {"H2": 0, "LiH": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", default="H2", choices=sorted(GRIDS))
    parser.add_argument("--optimizer", default="L-BFGS-B")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    name = args.molecule
    lengths = GRIDS[name]
    if args.quick:
        lengths = lengths[::4]

    print(f"{name} dissociation, UCCSD/{args.optimizer}, {len(lengths)} geometries\n")
    print(f"{'R (A)':>7} {'HF':>13} {'FCI':>13} {'VQE':>13} "
          f"{'VQE-FCI':>11} {'corr %':>8} {'iters':>6}")

    rows = []
    start = time.perf_counter()
    for length in lengths:
        molecule = load_molecule(
            name, bond_length=float(length), frozen_core=FROZEN[name]
        )
        hamiltonian = fermionic_hamiltonian(molecule)
        ansatz = UCCSDAnsatz(molecule)
        driver = VQE(hamiltonian, ansatz, optimizer=args.optimizer, max_iterations=400)
        result = driver.run(
            reference_energy=molecule.fci_energy,
            hartree_fock_energy=molecule.hf_energy,
        )
        rows.append({
            "bond_length": float(length),
            "hf": molecule.hf_energy,
            "fci": molecule.fci_energy,
            "vqe": result.energy,
            "error": result.error,
            "correlation_recovered": result.correlation_recovered,
            "chemically_accurate": bool(result.chemically_accurate),
            "iterations": result.iterations,
            "evaluations": result.evaluations,
            "num_qubits": molecule.num_qubits,
            "num_parameters": ansatz.num_parameters,
        })
        print(
            f"{length:>7.3f} {molecule.hf_energy:>13.8f} {molecule.fci_energy:>13.8f} "
            f"{result.energy:>13.8f} {result.error:>+11.2e} "
            f"{result.correlation_recovered * 100:>7.2f}% {result.iterations:>6}",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    errors = np.array([abs(r["error"]) for r in rows])
    accurate = sum(r["chemically_accurate"] for r in rows)
    print(f"\n{len(rows)} geometries in {elapsed:.1f}s")
    print(f"max |error| = {errors.max():.2e} Ha, mean = {errors.mean():.2e} Ha")
    print(f"within chemical accuracy: {accurate}/{len(rows)}")

    # How badly does Hartree-Fock do where it matters?
    worst = max(rows, key=lambda r: r["hf"] - r["fci"])
    print(f"largest HF error: {worst['hf'] - worst['fci']:.4f} Ha "
          f"at R = {worst['bond_length']} A "
          f"({(worst['hf'] - worst['fci']) / CHEMICAL_ACCURACY:.0f}x chemical accuracy)")

    out = RESULTS / (f"dissociation_{name}{'_quick' if args.quick else ''}.json")
    out.write_text(json.dumps({
        "molecule": name, "rows": rows, "seconds": elapsed,
        "optimizer": args.optimizer, "frozen_core": FROZEN[name],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
