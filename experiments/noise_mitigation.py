"""Noise, and how much of it zero-noise extrapolation can remove.

A noisy expectation value is *biased*, not just noisy: errors drag the state
towards the maximally mixed one, which pulls the measured energy away from
the truth by far more than chemical accuracy at realistic gate error rates.
Averaging harder does not fix a bias.

Zero-noise extrapolation makes the noise deliberately *worse* in a
controlled way -- by unitary folding, ``C -> C (C^dag C)^n``, which is the
identity mathematically but multiplies the accumulated error -- and
extrapolates back to zero.

    python experiments/noise_mitigation.py [--trajectories N] [--quick]
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
from vqelab.estimator import StatevectorEstimator
from vqelab.mitigation import ReadoutMitigator, fold_global, zero_noise_extrapolation
from vqelab.noise import DepolarizingNoise, TrajectoryEstimator
from vqelab.vqe import CHEMICAL_ACCURACY, VQE

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=6000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--readout-only", action="store_true",
                        help="reuse the stored ZNE data and redo only readout")
    args = parser.parse_args()
    trajectories = 300 if args.quick else args.trajectories
    rates = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    scales = (1.0, 3.0, 5.0)

    molecule = load_molecule("H2")
    hamiltonian = fermionic_hamiltonian(molecule)
    ansatz = UCCSDAnsatz(molecule)
    optimum = VQE(hamiltonian, ansatz).run(reference_energy=molecule.fci_energy)
    circuit = ansatz.circuit(optimum.parameters)
    ideal = StatevectorEstimator(hamiltonian).energy(circuit)

    print(f"H2/STO-3G at the VQE optimum: {ideal:.10f} Ha")
    print(f"circuit: {circuit.size} gates, {circuit.num_two_qubit_gates} two-qubit, "
          f"depth {circuit.depth()}")
    print(f"{trajectories} trajectories per point, folding scales {scales}\n")

    # Folding must not change the ideal answer -- check, do not assume.
    for scale in scales:
        folded = StatevectorEstimator(hamiltonian).energy(fold_global(circuit, int(scale)))
        assert abs(folded - ideal) < 1e-9, "folding changed the noiseless energy"
    print("folding verified as the identity in the absence of noise\n")

    stored = None
    if args.readout_only:
        stored = json.loads((RESULTS / "noise.json").read_text())
        rates = []

    rows = []
    print(f"{'2q rate':>9} {'raw error':>11} | "
          + " | ".join(f"{m:>22}" for m in ("linear", "richardson", "exponential")))
    start = time.perf_counter()
    for rate in rates:
        model = DepolarizingNoise(one_qubit=rate / 10, two_qubit=rate)
        estimator = TrajectoryEstimator(hamiltonian, model, trajectories, seed=1)

        energies, stderrs = [], []
        for scale in scales:
            energies.append(estimator.energy(fold_global(circuit, int(scale))))
            stderrs.append(estimator.last_stderr)

        lookup = dict(zip(scales, energies))
        entry = {
            "two_qubit_rate": rate,
            "scales": list(scales),
            "energies": energies,
            "stderrs": stderrs,
            "raw_error": energies[0] - ideal,
            "models": {},
        }
        cells = []
        for name in ("linear", "richardson", "exponential"):
            result = zero_noise_extrapolation(
                lambda s: lookup[s], scales, model=name, stderrs=stderrs
            )
            improvement = result.improvement(ideal)
            entry["models"][name] = {
                "mitigated": result.mitigated,
                "error": result.mitigated - ideal,
                **improvement,
            }
            cells.append(
                f"{result.mitigated:+.5f} ({improvement['factor']:>5.1f}x)"
            )
        rows.append(entry)
        print(f"{rate:>9.0e} {entry['raw_error']:>+11.2e} | "
              + " | ".join(f"{c:>22}" for c in cells), flush=True)

    elapsed = time.perf_counter() - start
    if stored is not None:
        rows, elapsed = stored["zne"], stored["seconds"]
    else:
        print(f"\n{len(rows) * len(scales)} noisy evaluations in {elapsed:.1f}s")

    # --- readout mitigation -------------------------------------------
    # Two regimes, because they say different things.  With exact
    # probabilities the inversion is exact and the result is uninterestingly
    # perfect.  With finite shots the inverse *amplifies* sampling noise,
    # which is the honest limit on how well this works in practice.
    print("\nreadout error mitigation (4 qubits, exact calibration matrix)")
    print(f"{'p':>6} {'exact: before':>14} {'after':>10} | "
          f"{'8192 shots: before':>19} {'after':>10}")
    readout_rows = []
    true_probabilities = np.abs(circuit.run().statevector.data) ** 2
    shot_rng = np.random.default_rng(7)
    shots = 8192
    for error in (0.01, 0.03, 0.05, 0.10):
        mitigator = ReadoutMitigator(molecule.num_qubits, error, seed=0)
        observed = mitigator.apply_noise(true_probabilities)
        recovered = mitigator.mitigate(observed)
        exact_before = float(np.abs(observed - true_probabilities).sum()) / 2
        exact_after = float(np.abs(recovered - true_probabilities).sum()) / 2

        # Same thing, but the observed distribution is a finite sample.
        sampled_before, sampled_after = [], []
        for _ in range(20):
            counts = shot_rng.multinomial(shots, observed) / shots
            fixed = mitigator.mitigate(counts)
            sampled_before.append(float(np.abs(counts - true_probabilities).sum()) / 2)
            sampled_after.append(float(np.abs(fixed - true_probabilities).sum()) / 2)
        shot_before = float(np.mean(sampled_before))
        shot_after = float(np.mean(sampled_after))

        readout_rows.append({
            "readout_error": error,
            "total_variation_before": exact_before,
            "total_variation_after": exact_after,
            "shots": shots,
            "sampled_before": shot_before,
            "sampled_after": shot_after,
        })
        print(f"{error:>6.2f} {exact_before:>14.4f} {exact_after:>10.1e} | "
              f"{shot_before:>19.4f} {shot_after:>10.4f}")

    out = RESULTS / ("noise_quick.json" if args.quick else "noise.json")
    out.write_text(json.dumps({
        "ideal": ideal,
        "fci": molecule.fci_energy,
        "chemical_accuracy": CHEMICAL_ACCURACY,
        "trajectories": trajectories,
        "circuit": {
            "gates": circuit.size,
            "two_qubit": circuit.num_two_qubit_gates,
            "depth": circuit.depth(),
        },
        "zne": rows,
        "readout": readout_rows,
        "seconds": elapsed,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
