"""Figures for the VQE experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
INK = "#1b1b1f"
GRID = "#d8d8de"
BLUE, ORANGE, GREEN, PURPLE = "#2f6fdb", "#d4541f", "#2c8c63", "#8a4fc4"
CHEMICAL_ACCURACY = 1.6e-3


def style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=10.5, color=INK, pad=9, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    if legend:
        ax.legend(frameon=False, fontsize=8.5)


def plot_dissociation() -> None:
    available = [m for m in ("H2", "LiH") if (RESULTS / f"dissociation_{m}.json").exists()]
    if not available:
        return
    fig, axes = plt.subplots(2, len(available), figsize=(6 * len(available), 7.6),
                             squeeze=False, gridspec_kw={"height_ratios": [2.1, 1]})
    fig.patch.set_facecolor("white")

    for column, molecule in enumerate(available):
        data = json.loads((RESULTS / f"dissociation_{molecule}.json").read_text())
        rows = data["rows"]
        r = np.array([x["bond_length"] for x in rows])
        hf = np.array([x["hf"] for x in rows])
        fci = np.array([x["fci"] for x in rows])
        vqe = np.array([x["vqe"] for x in rows])

        ax = axes[0][column]
        ax.plot(r, hf, "s--", color=ORANGE, ms=4, lw=1.6, label="Hartree–Fock")
        ax.plot(r, fci, "-", color=INK, lw=2.4, alpha=0.35, label="FCI (exact)")
        ax.plot(r, vqe, "o", color=BLUE, ms=5, lw=0, label="VQE (UCCSD)")
        qubits = rows[0]["num_qubits"]
        style(ax, f"{molecule} dissociation — {qubits} qubits",
              "bond length (Å)", "energy (Ha)")

        ax = axes[1][column]
        ax.semilogy(r, np.abs(hf - fci), "s--", color=ORANGE, ms=4, lw=1.6,
                    label="|HF − FCI|")
        ax.semilogy(r, np.maximum(np.abs(vqe - fci), 1e-16), "o-", color=BLUE,
                    ms=4, lw=1.6, label="|VQE − FCI|")
        ax.axhline(CHEMICAL_ACCURACY, color=GREEN, lw=1.4, ls=":",
                   label="chemical accuracy")
        style(ax, "Error against the exact answer", "bond length (Å)", "|error| (Ha)")

    fig.suptitle(
        "Hartree–Fock breaks down as bonds stretch; UCCSD tracks the exact curve",
        fontsize=12, color=INK, x=0.012, ha="left", y=0.985,
    )
    fig.tight_layout(pad=2.0, rect=(0, 0, 1, 0.955))
    out = RESULTS / "dissociation.png"
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


def plot_noise() -> None:
    path = RESULTS / "noise.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    ideal = data["ideal"]
    rows = data["zne"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.patch.set_facecolor("white")

    # --- extrapolation at one representative rate ----------------------
    ax = axes[0]
    chosen = min(rows, key=lambda r: abs(r["two_qubit_rate"] - 1e-3))
    scales = np.array(chosen["scales"])
    energies = np.array(chosen["energies"])
    ax.errorbar(scales, energies, yerr=chosen["stderrs"], marker="o", ms=6,
                lw=0, elinewidth=1.4, capsize=3, color=ORANGE, label="measured")
    grid = np.linspace(0, scales.max() * 1.05, 200)
    for name, colour in (("linear", BLUE), ("richardson", GREEN), ("exponential", PURPLE)):
        fit = chosen["models"][name]
        ax.plot([0], [fit["mitigated"]], "*", ms=13, color=colour,
                label=f"{name}: {fit['mitigated']:.4f}")
    ax.axhline(ideal, color=INK, lw=1.4, ls="--", label=f"ideal {ideal:.4f}")
    ax.set_xlim(left=-0.3)
    style(ax, f"Zero-noise extrapolation (2-qubit error {chosen['two_qubit_rate']:.0e})",
          "noise scale λ (unitary folding)", "energy (Ha)")

    # --- error vs noise rate, before and after -------------------------
    ax = axes[1]
    rates = np.array([r["two_qubit_rate"] for r in rows])
    ax.loglog(rates, [abs(r["raw_error"]) for r in rows], "s-", color=ORANGE,
              ms=5, lw=1.8, label="unmitigated")
    for name, colour in (("linear", BLUE), ("richardson", GREEN), ("exponential", PURPLE)):
        ax.loglog(rates, [abs(r["models"][name]["error"]) for r in rows], "o-",
                  color=colour, ms=4, lw=1.5, label=name)
    ax.axhline(CHEMICAL_ACCURACY, color=INK, lw=1.3, ls=":", label="chemical accuracy")
    style(ax, "Energy error before and after mitigation",
          "two-qubit gate error rate", "|error| (Ha)")

    # --- readout mitigation --------------------------------------------
    ax = axes[2]
    readout = data["readout"]
    errors = [r["readout_error"] for r in readout]
    index = np.arange(len(errors))
    shots = readout[0].get("shots", 8192)
    ax.bar(index - 0.2, [r.get("sampled_before", r["total_variation_before"])
                         for r in readout], 0.4, color=ORANGE, label="raw")
    ax.bar(index + 0.2, [max(r.get("sampled_after", 1e-17), 1e-17)
                         for r in readout], 0.4, color=BLUE,
           label=f"mitigated ({shots} shots)")
    ax.plot(index + 0.2, [max(r["total_variation_after"], 1e-17) for r in readout],
            "*", ms=11, color=GREEN, label="mitigated (exact statistics)")
    ax.set_yscale("log")
    ax.set_xticks(index)
    ax.set_xticklabels([f"{e:.0%}" for e in errors])
    style(ax, "Readout mitigation by calibration-matrix inversion",
          "per-qubit readout error", "distribution error", legend=False)
    # The empty band between the sampled bars and the exact-statistics stars
    # is the only place a legend does not collide with data.
    ax.legend(frameon=False, fontsize=8, loc="center left")

    fig.tight_layout(pad=2.0)
    out = RESULTS / "noise.png"
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_dissociation()
    plot_noise()
